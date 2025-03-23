#!/usr/bin/env python3
"""
Benchmark script for comparing docling's DocumentConverter performance
"""

from time import perf_counter
import os
from pathlib import Path
import logging
import json
import glob
import statistics
import argparse
import shutil
import concurrent.futures
from docling.document_converter import DocumentConverter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_file(file_path):
    """Process a single file using docling's DocumentConverter."""
    filename = os.path.basename(file_path)
    converter = DocumentConverter()

    try:
        start_time = perf_counter()
        result = converter.convert(file_path)
        end_time = perf_counter()

        # Extract relevant information from result
        content = {
            "success": True,
            "elements": result.model_dump(),
            "metadata": {
                "parsing_duration": (end_time - start_time) * 1000  # Convert to ms
            },
        }

        logger.info(f"Successfully processed: {filename}")
        return filename, json.dumps(content)
    except Exception as e:
        logger.error(f"Exception processing {filename}: {str(e)}")
        return filename, json.dumps({"success": False, "error": str(e)})


def analyze_parsing_results(
    processing_time_s: float,
    results_dir="/tmp/docling_responses",
):
    """Analyze parsing results and generate statistics."""
    stats = {
        "total_documents": 0,
        "total_elements": 0,
        "parsing_durations_ms": [],
        "elements_per_doc": [],
    }

    # Process all JSON files
    for json_file in glob.glob(f"{results_dir}/*.pdf.json"):
        with open(json_file) as f:
            try:
                response = json.load(f)
                if not response.get("success"):
                    continue

                # Get parsing duration
                parsing_duration_ms = response["metadata"]["parsing_duration"]

                # Update statistics
                stats["total_documents"] += 1
                stats["parsing_durations_ms"].append(parsing_duration_ms)

            except Exception as e:
                print(f"Error processing {json_file}: {e}")
                continue

    # Calculate aggregate statistics
    if stats["total_documents"] > 0:
        results = {
            "Total Documents Processed": stats["total_documents"],
            "Average Processing Time": f"{statistics.mean(stats['parsing_durations_ms']):.2f}ms",
            "Median Processing Time": f"{statistics.median(stats['parsing_durations_ms']):.2f}ms",
            "Documents per Second": stats["total_documents"] / processing_time_s,
            "Min Processing Time": f"{min(stats['parsing_durations_ms']):.2f}ms",
            "Max Processing Time": f"{max(stats['parsing_durations_ms']):.2f}ms",
        }

        # Print results
        print("\nParsing Statistics:")
        print("==================")
        for key, value in results.items():
            print(f"{key}: {value}")

        return results
    else:
        print("No valid documents found to analyze")
        return None


def process_directory(
    input_dir, max_concurrent=4, output_dir="/tmp/docling_responses", limit=None
):
    """Process all PDF files in the directory with concurrent processing."""
    input_path = Path(input_dir)
    pdf_files = list(input_path.glob("*.pdf"))
    if limit is not None:
        pdf_files = pdf_files[:limit]

    if not pdf_files:
        logger.warning(f"No PDF files found in {input_dir}")
        return

    # Create output directory
    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Storing responses in: {output_dir}")

    # Process files concurrently
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_concurrent) as executor:
        results = list(executor.map(process_file, pdf_files))

        # Save results
        for filename, content in results:
            if content:
                output_file = output_dir / f"{filename}.json"
                with open(output_file, "w") as f:
                    f.write(content)


def main():
    parser = argparse.ArgumentParser(
        description="Process PDF files using docling's DocumentConverter"
    )
    parser.add_argument("input_dir", help="Directory containing PDF files to process")
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=4,
        help="Maximum number of concurrent processes (default: 4)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit the number of PDF files to process (default: process all files)",
    )

    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        logger.error(f"Input directory '{args.input_dir}' does not exist")
        return

    # Run benchmark
    s = perf_counter()
    process_directory(args.input_dir, args.max_concurrent, limit=args.limit)
    logger.info("All files processed.")
    e = perf_counter()
    analyze_parsing_results(e - s)


if __name__ == "__main__":
    main()
