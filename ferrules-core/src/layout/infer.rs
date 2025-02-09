use ort::{
    ortsys,
    session::{
        NoSelectedOutputs, RunOptions, Session, SessionInputValue, SessionInputs,
        SharedSessionInner,
    },
    sys::{OrtStatus, OrtValue},
    value::{DynValue, Tensor, Value},
    AsPointer, ErrorCode,
};

use std::{
    cell::UnsafeCell,
    ffi::{c_char, c_void, CStr, CString},
    future::Future,
    ops::Deref,
    pin::Pin,
    ptr::NonNull,
    sync::{Arc, Mutex},
    task::{Context, Poll, Waker},
};

#[derive(Debug)]
pub struct SessionOutputs<'s> {
    values: Vec<DynValue>,
    effective_len: usize,
    backing_ptr: Option<(&'s ort::memory::Allocator, *mut ())>,
}

impl<'s> SessionOutputs<'s> {
    pub(crate) fn new(output_values: Vec<DynValue>) -> Self {
        Self {
            effective_len: output_values.len(),
            values: output_values,
            backing_ptr: None,
        }
    }
}
#[derive(Debug)]
pub(crate) struct InferenceFutInner<'s> {
    value: UnsafeCell<Option<anyhow::Result<SessionOutputs<'s>>>>,
    waker: Mutex<Option<Waker>>,
}

impl<'s> InferenceFutInner<'s> {
    pub(crate) fn new() -> Self {
        InferenceFutInner {
            waker: Mutex::new(None),
            value: UnsafeCell::new(None),
        }
    }

    pub(crate) fn try_take(&self) -> Option<anyhow::Result<SessionOutputs<'s>>> {
        unsafe { &mut *self.value.get() }.take()
    }

    pub(crate) fn emplace_value(&self, value: anyhow::Result<SessionOutputs<'s>>) {
        unsafe { &mut *self.value.get() }.replace(value);
    }

    pub(crate) fn set_waker(&self, waker: Option<&Waker>) {
        *self.waker.lock().expect("Poisoned waker mutex") = waker.map(|c| c.to_owned());
    }

    pub(crate) fn wake(&self) {
        if let Some(waker) = self.waker.lock().expect("Poisoned waker mutex").take() {
            waker.wake();
        }
    }
}

pub struct InferenceFut<'s> {
    inner: Arc<InferenceFutInner<'s>>,
    did_receive: bool,
}

unsafe impl Send for InferenceFutInner<'_> {}
unsafe impl Sync for InferenceFutInner<'_> {}

impl<'s> InferenceFut<'s> {
    pub(crate) fn new(inner: Arc<InferenceFutInner<'s>>) -> Self {
        Self {
            inner,
            did_receive: false,
        }
    }
}

impl<'s> Future for InferenceFut<'s> {
    type Output = anyhow::Result<SessionOutputs<'s>>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = Pin::into_inner(self);

        if let Some(v) = this.inner.try_take() {
            this.did_receive = true;
            return Poll::Ready(v);
        }

        this.inner.set_waker(Some(cx.waker()));
        Poll::Pending
    }
}

impl Drop for InferenceFut<'_> {
    fn drop(&mut self) {
        if !self.did_receive {
            self.inner.set_waker(None);
        }
    }
}

pub(crate) struct AsyncInferenceContext<'s> {
    pub(crate) inner: Arc<InferenceFutInner<'s>>,
    pub(crate) input_values: Vec<Tensor<f32>>,
    pub(crate) input_ort_values: Vec<*const OrtValue>,
    pub(crate) output_value_ptrs: Vec<*mut OrtValue>,
    pub(crate) session_inner: Arc<SharedSessionInner>,
    pub(crate) buffer_pool: Arc<BufferPool>,
}

pub(crate) fn char_p_to_string(raw: *const c_char) -> anyhow::Result<String> {
    if raw.is_null() {
        return Ok(String::new());
    }
    let c_string = unsafe { CStr::from_ptr(raw.cast_mut()).to_owned() };
    Ok(c_string.to_string_lossy().to_string())
}

pub(crate) fn status_to_result(status: *mut OrtStatus) -> anyhow::Result<()> {
    if status.is_null() {
        Ok(())
    } else {
        let _code = ErrorCode::from(ortsys![unsafe GetErrorCode(status)]);
        let raw: *const std::os::raw::c_char = ortsys![unsafe GetErrorMessage(status)];
        match char_p_to_string(raw) {
            Ok(msg) => {
                ortsys![unsafe ReleaseStatus(status)];
                anyhow::bail!(msg)
            }
            Err(err) => {
                ortsys![unsafe ReleaseStatus(status)];
                anyhow::bail!(format!("(failed to convert UTF-8: {err})"))
            }
        }
    }
}
pub unsafe extern "C" fn async_callback(
    user_data: *mut c_void,
    _: *mut *mut OrtValue,
    _: usize,
    status: *mut OrtStatus,
) {
    let ctx = unsafe { Box::from_raw(user_data.cast::<AsyncInferenceContext<'_>>()) };

    if let Err(e) = status_to_result(status) {
        ctx.inner.emplace_value(Err(e));
        ctx.inner.wake();
        return;
    }

    let outputs: Vec<Value> = ctx
        .output_value_ptrs
        .into_iter()
        .map(|tensor_ptr| unsafe {
            Value::from_ptr(
                NonNull::new(tensor_ptr)
                    .expect("OrtValue ptr returned from session Run should not be null"),
                Some(Arc::clone(&ctx.session_inner)),
            )
        })
        .collect();

    ctx.inner.emplace_value(Ok(SessionOutputs::new(outputs)));
    ctx.inner.wake();
}

struct PooledSessionInner {
    session: Arc<SharedSessionInner>,
    buffer_pool: Arc<BufferPool>,
}

struct BufferPool {
    store: Mutex<Vec<Tensor<f32>>>,
    cvar: std::sync::Condvar,
}
impl BufferPool {
    fn put(&self, buffer: Tensor<f32>) -> anyhow::Result<()> {
        let mut store = self.store.lock().expect("poison lock");
        store.push(buffer);
        self.cvar.notify_one();
        Ok(())
    }
    fn get(&self) -> Tensor<f32> {
        let mut store = self.store.lock().unwrap();
        loop {
            let result = store.pop();
            match result {
                Some(t) => return t,
                None => {
                    store = self.cvar.wait(store).unwrap();
                }
            }
        }
    }
}

impl PooledSessionInner {
    pub(crate) fn infer_async_inner<'s>(
        &mut self,
        input_names: &[String],
        output_names: &[String],
        input_values: Vec<Tensor<f32>>,
    ) -> anyhow::Result<InferenceFut<'s>> {
        let input_name_ptrs: Vec<*const c_char> = input_names
            .iter()
            .map(|n| CString::new(n.as_bytes()).unwrap_or_else(|_| unreachable!()))
            .map(|n| n.into_raw().cast_const())
            .collect();

        let output_name_ptrs: Vec<*const c_char> = output_names
            .iter()
            .map(|n| CString::new(n.as_bytes()).unwrap_or_else(|_| unreachable!()))
            .map(|n| n.into_raw().cast_const())
            .collect();
        let output_tensor_ptrs: Vec<*mut OrtValue> = vec![std::ptr::null_mut(); output_names.len()];

        let input_ort_values: Vec<*const OrtValue> = input_values
            .iter()
            .map(|input_array_ort| input_array_ort.ptr())
            .collect();

        let async_inner = Arc::new(InferenceFutInner::new());

        let ctx = Box::leak(Box::new(AsyncInferenceContext {
            inner: Arc::clone(&async_inner),
            input_values,
            input_ort_values,
            output_value_ptrs: output_tensor_ptrs,
            session_inner: Arc::clone(&self.session),
            buffer_pool: Arc::clone(&self.buffer_pool),
        }));
        let run_opts = Arc::new(unsafe {
            // SAFETY: transmuting from `RunOptions<NoSelectedOutputs>` to `RunOptions<O>`; safe because its just a marker
            std::mem::transmute::<RunOptions<NoSelectedOutputs>, RunOptions<NoSelectedOutputs>>(
                RunOptions::new()?,
            )
        });
        ortsys![
            unsafe RunAsync(
                self.session.ptr() as *mut _ ,
                run_opts.ptr(),
                input_name_ptrs.as_ptr(),
                ctx.input_ort_values.as_ptr(),
                ctx.input_ort_values.len(),
                output_name_ptrs.as_ptr(),
                output_name_ptrs.len(),
                ctx.output_value_ptrs.as_mut_ptr(),
                Some(async_callback),
                ctx as *mut _ as *mut c_void
            )?
        ];

        Ok(InferenceFut::new(async_inner))
    }
}
