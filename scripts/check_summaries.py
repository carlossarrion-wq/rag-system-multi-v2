#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check S3 summaries and documents for a specific application.
This script provides a quick overview of document and summary status.
"""

import json
import boto3
import sys
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.multi_app_config_manager import MultiAppConfigManager

def check_summaries(app_name):
    """
    Check S3 summaries and documents for a specific application.
    
    Args:
        app_name: Application name to check
    """
    try:
        # Load config
        config_manager = MultiAppConfigManager()
        
        # Validate application
        if not config_manager.validate_application(app_name):
            available_apps = config_manager.get_available_applications()
            print(f'Error: Application "{app_name}" not found.')
            print(f'Available applications: {", ".join(available_apps)}')
            return False
        
        # Get application config
        app_config = config_manager.get_application_config(app_name)
        s3_config = app_config['services']['s3']
        
        # Initialize S3 client
        s3_client = boto3.client('s3')
        bucket = s3_config['bucket']
        documents_prefix = s3_config['documents_prefix']
        
        print(f'=== CHECKING APPLICATION: {app_name.upper()} ===')
        print(f'S3 Bucket: {bucket}')
        print(f'Documents Prefix: {documents_prefix}')
        print()
        
        # Check S3 summaries
        print('=== S3 SUMMARIES ===')
        # Use the same structure as the main ingestion system
        summaries_prefix = f'applications/{app_name}/summaries/'
        try:
            response = s3_client.list_objects_v2(Bucket=bucket, Prefix=summaries_prefix)
            if 'Contents' in response:
                summary_count = len(response['Contents'])
                print(f'Found {summary_count} summary files:')
                for obj in response['Contents']:
                    print(f'  - {obj["Key"]}')
            else:
                print('No summary files found')
                summary_count = 0
        except Exception as e:
            print(f'Error listing summaries: {e}')
            summary_count = 0
        
        # Check S3 documents
        print('\n=== S3 DOCUMENTS ===')
        try:
            response = s3_client.list_objects_v2(Bucket=bucket, Prefix=documents_prefix)
            if 'Contents' in response:
                # Filter out directories and non-document files
                doc_files = []
                for obj in response['Contents']:
                    key = obj['Key']
                    if not key.endswith('/') and any(key.lower().endswith(ext) for ext in ['.pdf', '.docx', '.txt', '.md', '.xml', '.xlsx', '.xls', '.jpg', '.jpeg', '.png']):
                        doc_files.append(obj)
                
                doc_count = len(doc_files)
                print(f'Found {doc_count} document files:')
                for obj in doc_files:
                    print(f'  - {obj["Key"]}')
            else:
                print('No document files found')
                doc_count = 0
        except Exception as e:
            print(f'Error listing documents: {e}')
            doc_count = 0
        
        # Summary
        print(f'\n=== SUMMARY FOR {app_name.upper()} ===')
        print(f'Documents: {doc_count}')
        print(f'Summaries: {summary_count}')
        print(f'Missing summaries: {max(0, doc_count - summary_count)}')
        
        if doc_count > 0:
            coverage_percent = (summary_count / doc_count) * 100
            print(f'Summary coverage: {coverage_percent:.1f}%')
        
        return True
        
    except Exception as e:
        print(f'Error checking summaries for {app_name}: {e}')
        return False

def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description='Check S3 summaries and documents for a specific application',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 scripts/check_summaries.py gadea
  python3 scripts/check_summaries.py pds
  python3 scripts/check_summaries.py --app gadea
        """
    )
    
    parser.add_argument(
        'app_name',
        nargs='?',
        help='Application name to check (e.g., gadea, pds)'
    )
    
    parser.add_argument(
        '--app',
        help='Alternative way to specify application name'
    )
    
    args = parser.parse_args()
    
    # Determine app name from arguments
    app_name = args.app_name or args.app
    
    if not app_name:
        parser.print_help()
        print('\nError: Application name is required')
        return 1
    
    # Check summaries for the specified application
    success = check_summaries(app_name)
    return 0 if success else 1

if __name__ == '__main__':
    sys.exit(main())
