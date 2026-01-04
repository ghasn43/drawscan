#!/usr/bin/env python3
"""
Main DXF BOQ Extractor Application
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
from dxf_processor import DXFProcessor
from boq_generator import BOQGenerator


def main():
    """Main application function"""
    parser = argparse.ArgumentParser(
        description='DXF BOQ Extractor - Extract engineering quantities from DXF files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Process all DXF files in input folder
  %(prog)s --units=mm         # Assume drawings are in millimeters
  %(prog)s --project="My Building"  # Set project name
  %(prog)s --input=my_files --output=results  # Custom folders
        """
    )
    
    parser.add_argument('--input', '-i', default='input', 
                       help='Input folder with DXF files (default: input)')
    parser.add_argument('--output', '-o', default='output', 
                       help='Output folder for results (default: output)')
    parser.add_argument('--units', '-u', default='m', 
                       choices=['mm', 'cm', 'm', 'inch', 'ft'],
                       help='Drawing units (default: m for meters)')
    parser.add_argument('--project', '-p', default='DXF Engineering Project', 
                       help='Project name for reports')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show detailed processing information')
    parser.add_argument('--list-only', '-l', action='store_true',
                       help='List files without processing')
    
    args = parser.parse_args()
    
    print("╔══════════════════════════════════════════════════════════════════════════════╗")
    print("║                    DXF BOQ EXTRACTOR - PROFESSIONAL EDITION                   ║")
    print("╚══════════════════════════════════════════════════════════════════════════════╝")
    print()
    
    # Setup folders
    input_folder = Path(args.input)
    output_folder = Path(args.output)
    
    input_folder.mkdir(exist_ok=True, parents=True)
    output_folder.mkdir(exist_ok=True, parents=True)
    
    # Find DXF files
    dxf_files = list(input_folder.glob("*.dxf")) + list(input_folder.glob("*.DXF"))
    
    if not dxf_files:
        print(f"❌ No DXF files found in: {input_folder.absolute()}")
        print()
        print("Please place your DXF files in one of these locations:")
        print(f"  1. {input_folder.absolute()}")
        print(f"  2. Or specify a different folder: python main_app.py --input=your_folder")
        print()
        print("Supported DXF formats:")
        print("  • AutoCAD DXF (all versions)")
        print("  • LibreCAD DXF")
        print("  • Other CAD software DXF exports")
        print()
        input("Press Enter to exit...")
        return
    
    print(f"📁 Found {len(dxf_files)} DXF file(s) in '{input_folder.name}' folder:")
    for i, file in enumerate(dxf_files, 1):
        size_kb = file.stat().st_size / 1024
        print(f"   {i:2d}. {file.name:<30} ({size_kb:.1f} KB)")
    
    if args.list_only:
        print("\n✅ File listing complete. Use without --list-only to process files.")
        input("\nPress Enter to exit...")
        return
    
    print(f"\n⚙️  Settings:")
    print(f"   • Units: {args.units}")
    print(f"   • Project: {args.project}")
    print(f"   • Output: {output_folder.absolute()}")
    print()
    
    # Confirm processing
    if len(dxf_files) > 1:
        response = input(f"Process {len(dxf_files)} files? (y/n): ").lower()
        if response not in ['y', 'yes']:
            print("\nProcessing cancelled.")
            return
    
    print("━" * 80)
    
    # Process each file
    processed_count = 0
    for dxf_file in dxf_files:
        print(f"\n📄 PROCESSING: {dxf_file.name}")
        print("━" * 40)
        
        try:
            # Initialize processor
            processor = DXFProcessor(units=args.units)
            
            # Load DXF file
            print(f"🔍 Loading DXF file...")
            if not processor.load_file(dxf_file):
                print(f"❌ Failed to load {dxf_file.name}")
                continue
            
            # Get drawing information
            drawing_info = processor.get_drawing_info()
            print(f"✅ File loaded successfully")
            print(f"   • Units: {drawing_info['units'].upper()}")
            print(f"   • Entities: {drawing_info['entity_count']:,}")
            print(f"   • Layers: {drawing_info['layer_count']}")
            
            if 'extents' in drawing_info and drawing_info['extents']:
                ext = drawing_info['extents']
                print(f"   • Drawing size: {ext['width']:.2f} × {ext['height']:.2f} units")
            
            # Analyze layers
            print(f"\n🔍 Analyzing layers...")
            layers = processor.analyze_layers()
            active_layers = [l for l in layers if l.entity_count > 0]
            print(f"✅ Found {len(active_layers)} active layers (with entities)")
            
            if args.verbose and active_layers:
                print(f"\n   Active layers:")
                for layer in active_layers[:5]:  # Show first 5
                    main_entity = max(layer.entity_types.items(), key=lambda x: x[1])[0] if layer.entity_types else 'None'
                    print(f"   • {layer.name:<20} : {layer.entity_count:>4} entities ({layer.suggested_purpose})")
                
                if len(active_layers) > 5:
                    print(f"   ... and {len(active_layers) - 5} more")
            
            # Extract entities
            print(f"\n📐 Extracting geometry data...")
            entities = processor.extract_entities()
            
            if not entities:
                print(f"⚠️  No extractable entities found")
                continue
            
            # Show entity type summary
            entity_types = {}
            for entity in entities:
                entity_types[entity.entity_type] = entity_types.get(entity.entity_type, 0) + 1
            
            print(f"✅ Extracted {len(entities):,} entities")
            if entity_types:
                print(f"   Entity types: " + ", ".join([f"{k}({v})" for k, v in entity_types.items()]))
            
            # Calculate quantities
            print(f"\n🧮 Calculating quantities...")
            quantities = processor.calculate_quantities(entities)
            
            # Show important quantities
            print(f"✅ Quantity summary:")
            if quantities['walls']['length_m'] > 0:
                print(f"   • Walls: {quantities['walls']['length_m']:.1f} m length, {quantities['walls']['area_m2']:.1f} m² area")
            
            if quantities['slabs']['area_m2'] > 0:
                print(f"   • Slabs: {quantities['slabs']['area_m2']:.1f} m² area, {quantities['slabs']['volume_m3']:.1f} m³ volume")
            
            if quantities['beams']['length_m'] > 0:
                print(f"   • Beams: {quantities['beams']['length_m']:.1f} m length")
            
            if quantities['columns']['count'] > 0:
                print(f"   • Columns: {quantities['columns']['count']} nos, {quantities['columns']['volume_m3']:.1f} m³ volume")
            
            if quantities['doors']['count'] > 0:
                print(f"   • Doors: {quantities['doors']['count']} nos")
            
            if quantities['windows']['count'] > 0:
                print(f"   • Windows: {quantities['windows']['count']} nos")
            
            # Generate BOQ
            print(f"\n📋 Generating Bill of Quantities...")
            boq_generator = BOQGenerator(project_name=args.project)
            boq_df = boq_generator.generate_boq(quantities, processor)
            
            if boq_df.empty:
                print(f"⚠️  No BOQ items generated (no recognized elements)")
            else:
                item_count = len(boq_df) - 1  # Exclude total row
                total_amount = boq_df.iloc[-1]['Amount'] if not boq_df.empty else 0
                print(f"✅ Generated {item_count} BOQ items")
                print(f"   • Estimated cost: Rs.{total_amount:,.2f}")
            
            # Export results
            print(f"\n💾 Exporting results...")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = dxf_file.stem
            
            # Export to Excel
            excel_file = output_folder / f"{base_name}_BOQ_{timestamp}.xlsx"
            boq_generator.export_to_excel(
                boq_df, entities, layers, excel_file, drawing_info, quantities
            )
            print(f"   📊 Excel file: {excel_file.name}")
            
            # Export to JSON
            json_file = output_folder / f"{base_name}_data_{timestamp}.json"
            all_data = {
                'project': args.project,
                'filename': dxf_file.name,
                'timestamp': timestamp,
                'drawing_info': drawing_info,
                'layers': [layer.__dict__ for layer in layers],
                'entities': [entity.__dict__ for entity in entities],
                'quantities': quantities,
                'units': args.units
            }
            boq_generator.export_to_json(all_data, json_file)
            print(f"   📄 JSON data: {json_file.name}")
            
            # Create text report
            report_file = output_folder / f"{base_name}_report_{timestamp}.txt"
            boq_generator.create_report(boq_df, drawing_info, quantities, report_file)
            print(f"   📝 Text report: {report_file.name}")
            
            processed_count += 1
            print(f"\n✅ COMPLETED: {dxf_file.name}")
            print("━" * 40)
            
        except Exception as e:
            print(f"\n❌ ERROR processing {dxf_file.name}:")
            print(f"   {str(e)}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            print("━" * 40)
    
    # Final summary
    print("\n" + "="*80)
    print("PROCESSING SUMMARY")
    print("="*80)
    
    if processed_count > 0:
        print(f"✅ Successfully processed {processed_count} of {len(dxf_files)} file(s)")
        print(f"📁 Output folder: {output_folder.absolute()}")
        print()
        print("Generated files for each DXF:")
        print("  • Excel file (.xlsx)    - Complete BOQ with multiple sheets")
        print("  • JSON file (.json)     - Raw data for integration")
        print("  • Text report (.txt)    - Summary report")
        print()
        print("📊 Excel file contains:")
        print("  1. BOQ_Summary    - Bill of Quantities with rates")
        print("  2. Drawing_Info   - Drawing properties and statistics")
        print("  3. Layers         - Layer analysis and classification")
        print("  4. Quantities     - Calculated quantities summary")
        print("  5. Entities       - Detailed entity information")
    else:
        print("❌ No files were successfully processed")
    
    print()
    print("="*80)
    
    # Show output files
    if processed_count > 0 and output_folder.exists():
        output_files = list(output_folder.glob("*"))
        if output_files:
            print(f"\n📁 Generated {len(output_files)} file(s) in output folder:")
            for file in output_files[:10]:  # Show first 10 files
                size_mb = file.stat().st_size / (1024 * 1024)
                print(f"   • {file.name:<40} ({size_mb:.2f} MB)")
            
            if len(output_files) > 10:
                print(f"   ... and {len(output_files) - 10} more files")
    
    print()
    input("Press Enter to exit...")


if __name__ == "__main__":
    main()