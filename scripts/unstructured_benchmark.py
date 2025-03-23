#!/usr/bin/env python3
"""
UNSTRUCTURED:
    Parsing Statistics:
    ==================
    Total Documents Processed: 20
    Total Elements Extracted: 38303
    Average Elements per Document: 1915.15
    Average Processing Time: 65124.20ms
    Median Processing Time: 25282.43ms
    Documents per Second: 0.03701149072375816
    Min Processing Time: 2331.92ms
    Max Processing Time: 512820.16ms

FERRULES: (94x parsing )
    Parsing Statistics:
    ==================
    Total Documents Processed: 20
    Total Pages Processed: 525
    Total Blocks Extracted: 4991
    Average Pages per Document: 26.25
    Average Blocks per Document: 249.55
    Average Blocks per Page: 11.515839964904599
    Average Processing Time: 1920.50ms
    Median Processing Time: 1107.50ms
    Pages per Second: 74.31
    Documents per Second: 2.83
    Min Processing Time: 194.00ms
    Max Processing Time: 6949.00ms
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
from unstructured.partition.pdf import partition_pdf
import concurrent.futures

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_parsing_results(
    processing_time_s: float,
    results_dir="/tmp/unstructured_responses",
):
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

                # Get basic counts
                n_elements = len(response["elements"])
                parsing_duration_ms = response["metadata"]["parsing_duration"]

                # Update statistics
                stats["total_documents"] += 1
                stats["total_elements"] += n_elements
                stats["parsing_durations_ms"].append(parsing_duration_ms)
                stats["elements_per_doc"].append(n_elements)

            except Exception as e:
                print(f"Error processing {json_file}: {e}")
                continue

    # Calculate aggregate statistics
    if stats["total_documents"] > 0:
        results = {
            "Total Documents Processed": stats["total_documents"],
            "Total Elements Extracted": stats["total_elements"],
            "Average Elements per Document": statistics.mean(stats["elements_per_doc"]),
            "Average Processing Time": f"{statistics.mean(stats['parsing_durations_ms']):.2f}ms",
            "Median Processing Time": f"{statistics.median(stats['parsing_durations_ms']):.2f}ms",
            "Documents per Second": stats["total_documents"] / processing_time_s,
            "Min Processing Time": f"{min(stats['parsing_durations_ms']):.2f}ms",
            "Max Processing Time": f"{max(stats['parsing_durations_ms']):.2f}ms",
        }

        # Print results in a formatted way
        print("\nParsing Statistics:")
        print("==================")
        for key, value in results.items():
            print(f"{key}: {value}")

        return results
    else:
        print("No valid documents found to analyze")
        return None


def process_file(file_path):
    """Process a single file using unstructured with highres strategy."""
    filename = os.path.basename(file_path)

    try:
        start_time = perf_counter()
        elements = partition_pdf(file_path, strategy="hi_res", include_metadata=True)
        end_time = perf_counter()

        # Convert elements to serializable format
        elements_data = [
            {
                "text": str(elem.text),
                "type": type(elem).__name__,
                "metadata": elem.to_dict(),
            }
            for elem in elements
        ]

        result = {
            "success": True,
            "elements": elements_data,
            "metadata": {
                "parsing_duration": (end_time - start_time) * 1000  # Convert to ms
            },
        }

        logger.info(f"Successfully processed: {filename}")
        return filename, json.dumps(result)
    except Exception as e:
        logger.error(f"Exception processing {filename}: {str(e)}")
        return filename, json.dumps({"success": False, "error": str(e)})


def process_directory(
    input_dir, max_concurrent=4, output_dir="/tmp/unstructured_responses", limit=None
):
    """Process all PDF files in the directory with concurrent processing."""
    input_path = Path(input_dir)
    pdf_files = list(input_path.glob("*.pdf"))
    if limit is not None:
        pdf_files = pdf_files[:limit]

    if not pdf_files:
        logger.warning(f"No PDF files found in {input_dir}")
        return

    # Create temporary directory for responses
    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Storing responses in: {output_dir}")

    # Process files concurrently using ProcessPoolExecutor
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
        description="Process PDF files using unstructured."
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

    # Parse arguments
    args = parser.parse_args()

    # Verify input directory exists
    if not os.path.isdir(args.input_dir):
        logger.error(f"Input directory '{args.input_dir}' does not exist")
        return

    # Run the process
    s = perf_counter()
    process_directory(args.input_dir, args.max_concurrent, limit=args.limit)
    logger.info("All files processed.")
    e = perf_counter()
    analyze_parsing_results(e - s)


if __name__ == "__main__":
    main()
