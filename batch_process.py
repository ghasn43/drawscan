#!/usr/bin/env python3
"""
Batch processing for multiple DXF projects
"""

import json
from pathlib import Path
from datetime import datetime
import pandas as pd

def batch_process(config_file="batch_config.json"):
    """Process multiple projects from config file"""
    
    # Load configuration
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    results = []
    
    for project in config['projects']:
        print(f"\n{'='*60}")
        print(f"Processing: {project['name']}")
        print(f"{'='*60}")
        
        input_folder = Path(project['input_folder'])
        output_folder = Path(project['output_folder'])
        units = project.get('units', 'm')
        
        if not input_folder.exists():
            print(f"❌ Input folder not found: {input_folder}")
            continue
        
        # Process each DXF file
        dxf_files = list(input_folder.glob("*.dxf")) + list(input_folder.glob("*.DXF"))
        
        for dxf_file in dxf_files:
            try:
                # Import and use your existing processor
                from dxf_processor import DXFProcessor
                from boq_generator import BOQGenerator
                
                processor = DXFProcessor(units=units)
                if not processor.load_file(dxf_file):
                    continue
                
                drawing_info = processor.get_drawing_info()
                layers = processor.analyze_layers()
                entities = processor.extract_entities()
                quantities = processor.calculate_quantities(entities)
                
                # Generate BOQ
                generator = BOQGenerator(project_name=project['name'])
                boq_df = generator.generate_boq(quantities, processor)
                
                if not boq_df.empty:
                    # Save results
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    # Create project-specific output folder
                    project_output = output_folder / project['name'].replace(' ', '_')
                    project_output.mkdir(parents=True, exist_ok=True)
                    
                    # Export
                    excel_file = project_output / f"{dxf_file.stem}_{timestamp}.xlsx"
                    generator.export_to_excel(
                        boq_df, entities, layers, excel_file, drawing_info, quantities
                    )
                    
                    # Record summary
                    total_amount = boq_df.iloc[-1]['Amount'] if not boq_df.empty else 0
                    
                    results.append({
                        'project': project['name'],
                        'file': dxf_file.name,
                        'entities': len(entities),
                        'walls_length_m': quantities['walls']['length_m'],
                        'walls_area_m2': quantities['walls']['area_m2'],
                        'total_amount': total_amount,
                        'output_file': str(excel_file),
                        'timestamp': timestamp
                    })
                    
                    print(f"✅ {dxf_file.name}: {len(entities)} entities, Amount: {total_amount:,.2f}")
                    
            except Exception as e:
                print(f"❌ Error processing {dxf_file.name}: {e}")
    
    # Create summary report
    if results:
        summary_df = pd.DataFrame(results)
        summary_file = Path("batch_summary.xlsx")
        
        with pd.ExcelWriter(summary_file, engine='openpyxl') as writer:
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Group by project
            project_summary = summary_df.groupby('project').agg({
                'file': 'count',
                'entities': 'sum',
                'walls_length_m': 'sum',
                'walls_area_m2': 'sum',
                'total_amount': 'sum'
            }).reset_index()
            
            project_summary.to_excel(writer, sheet_name='Project_Summary', index=False)
        
        print(f"\n{'='*60}")
        print(f"✅ Batch processing complete!")
        print(f"📊 Summary saved to: {summary_file}")
        print(f"{'='*60}")
    
    return results

if __name__ == "__main__":
    # Create sample config if not exists
    config_file = "batch_config.json"
    if not Path(config_file).exists():
        sample_config = {
            "projects": [
                {
                    "name": "Club House",
                    "input_folder": "input/club_house",
                    "output_folder": "output",
                    "units": "m"
                },
                {
                    "name": "Residential Building",
                    "input_folder": "input/residential",
                    "output_folder": "output",
                    "units": "m"
                }
            ]
        }
        
        with open(config_file, 'w') as f:
            json.dump(sample_config, f, indent=2)
        
        print(f"Created sample config: {config_file}")
        print("Please edit the config file and run again.")
    else:
        batch_process(config_file)