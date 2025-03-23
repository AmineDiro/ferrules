#!/usr/bin/env python3

from unstructured.partition.pdf import partition_pdf

from docling.document_converter import DocumentConverter

file_path = (
    "/Users/amine/data/quivr/sample-knowledges/197c383c-b37f-45c7-97fd-fc3037b5d791.pdf"
)
# elements = partition_pdf(file_path, strategy="hi_res", include_metadata=True)

converter = DocumentConverter()
result = converter.convert(file_path)
breakpoint()
