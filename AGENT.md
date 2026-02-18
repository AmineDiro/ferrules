# AGENT
First remember to always use the python venv in python/.venv ( add this to AGENT.md so you rember)  when you want to run python scrips or install python packages. 

## Monitoring ANE Usage

To track Apple Neural Engine (ANE) usage, you can use `macmon` with the pipe command. This is useful for verifying if models are running on the ANE.

```bash
macmon pipe -s 10 -i 500
```

The `-s 10` flag sets the number of samples, and `-i 500` sets the interval in milliseconds.

Example output (JSON stream):
```json
{"temp":{"cpu_temp_avg":40.835796,"gpu_temp_avg":44.810947},"memory":{"ram_total":25769803776,"ram_usage":14942404608,"swap_total":6442450944,"swap_usage":4665114624},"ecpu_usage":[1210,0.08278669],"pcpu_usage":[1260,0.002436177],"gpu_usage":[169,0.0027520815],"cpu_power":0.06117144,"gpu_power":0.0012522695,"ane_power":0.0,"all_power":0.06242371,"sys_power":10.269848,"ram_power":0.0812975,"gpu_ram_power":0.0}
{"temp":{"cpu_temp_avg":40.835796,"gpu_temp_avg":44.810947},"memory":{"ram_total":25769803776,"ram_usage":15089975296,"swap_total":6442450944,"swap_usage":4665114624},"ecpu_usage":[1692,0.21567217],"pcpu_usage":[1364,0.0064955307],"gpu_usage":[338,0.01348789],"cpu_power":0.15673351,"gpu_power":0.03071117,"ane_power":0.0,"all_power":0.18744469,"sys_power":10.272432,"ram_power":0.14051929,"gpu_ram_power":0.0}
```
Look for `ane_power` > 0.0 to confirm usage.
