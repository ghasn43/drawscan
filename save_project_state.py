#!/usr/bin/env python3
"""
save_project_state.py - Save current project state for continuing in new chat
"""
import json
import os
import sys
from datetime import datetime
from pathlib import Path

def save_project_state():
    """Save complete project state"""
    
    print("Saving DXF BOQ Extractor Project State...")
    print("=" * 60)
    
    state = {
        'timestamp': datetime.now().isoformat(),
        'project_name': 'DXF BOQ Extractor Pro',
        'version': '2.0.0',
        'project_description': 'Professional tool for extracting Bill of Quantities from DXF drawings and comparing BOQs',
        'files_created': [],
        'last_achievements': [],
        'next_steps': [],
        'recent_fixes': [],
        'working_commands': [],
        'dependencies': []
    }
    
    # List all files in project directory
    project_dir = Path('.')
    project_files = []
    
    for file_path in project_dir.rglob('*'):
        if file_path.is_file():
            # Skip hidden files and __pycache__
            if any(part.startswith('.') or part == '__pycache__' for part in file_path.parts):
                continue
            
            rel_path = str(file_path.relative_to(project_dir))
            # Include important file types
            if any(rel_path.endswith(ext) for ext in ['.py', '.txt', '.md', '.json', '.dxf', '.xlsx', '.xls', '.html', '.png']):
                project_files.append(rel_path)
    
    state['files_created'] = sorted(project_files)
    
    # Save achievements
    state['last_achievements'] = [
        "Built complete DXF processing engine with unit detection",
        "Created BOQ generator with Excel/JSON/CSV export",
        "Developed command-line interface (main_app.py)",
        "Built comprehensive Streamlit web application",
        "Added visualization module with 2D/3D charts and dashboards",
        "Implemented BOQ Comparator for client-contractor reconciliation",
        "Fixed Python 3.14 compatibility issues",
        "Successfully processed real engineering drawings",
        "Added error handling and validation throughout",
        "Created comprehensive test suite"
    ]
    
    # Recent fixes
    state['recent_fixes'] = [
        "Added missing process_dxf() method to DXFProcessor class",
        "Fixed BOQGenerator to work with walls list instead of quantities dict",
        "Fixed Matplotlib compatibility issues in visualization.py",
        "Resolved Series hashing errors in boq_comparator.py",
        "Fixed column standardization issues in BOQ comparison",
        "Added robust error handling for file operations",
        "Fixed Excel export buffer issues"
    ]
    
    # Working commands
    state['working_commands'] = [
        "streamlit run streamlit_app.py",
        "streamlit run comparator_app.py",
        "python main_app.py",
        "python test_visualization.py",
        "python test_dxf_processing.py",
        "python boq_comparator.py"
    ]
    
    # Dependencies
    state['dependencies'] = [
        "streamlit>=1.28.0",
        "pandas>=2.0.0",
        "ezdxf>=1.0.0",
        "matplotlib>=3.7.0",
        "plotly>=5.17.0",
        "openpyxl>=3.1.0",
        "seaborn>=0.13.0",
        "numpy>=1.24.0"
    ]
    
    # Next steps
    state['next_steps'] = [
        "Add user authentication and authorization system",
        "Implement project database with SQLite/PostgreSQL",
        "Add PDF report generation using ReportLab",
        "Integrate cloud storage (Google Drive, Dropbox)",
        "Create REST API for integration with other software",
        "Optimize for mobile devices",
        "Add multi-language support",
        "Implement cost optimization algorithms",
        "Add version control for projects",
        "Create admin dashboard"
    ]
    
    # Save to JSON
    with open('project_state.json', 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Project state saved to 'project_state.json'")
    print(f"   Total files tracked: {len(state['files_created'])}")
    
    # Build summary content using string concatenation
    achievements_str = '\n'.join('✅ ' + achievement for achievement in state['last_achievements'])
    fixes_str = '\n'.join('🔧 ' + fix for fix in state['recent_fixes'])
    commands_str = '\n'.join(state['working_commands'])
    deps_str = '\n'.join(state['dependencies'])
    next_steps_str = '\n'.join('1. ' + step for step in state['next_steps'])
    
    summary_content = f"""# DXF BOQ Extractor Pro - Project Summary

## Project Information
- **Project Name**: {state['project_name']}
- **Version**: {state['version']}
- **Last Updated**: {state['timestamp']}
- **Description**: {state['project_description']}

## Project Structure
The project consists of {len(state['files_created'])} files:

### Core Modules:
- `dxf_processor.py` - DXF file processing and wall extraction
- `boq_generator.py` - Bill of Quantities generation with cost calculation
- `visualization.py` - 2D/3D visualizations and charts
- `boq_comparator.py` - BOQ comparison between client and contractor

### Applications:
- `streamlit_app.py` - Main web interface with both DXF processing and BOQ comparison
- `comparator_app.py` - Standalone BOQ comparison tool
- `main_app.py` - Command-line interface

### Supporting Files:
- `requirements.txt` - Python dependencies
- Test files (`test_*.py`)
- Sample data files

## Achievements
{achievements_str}

## Recent Fixes Applied
{fixes_str}

## Working Commands
```bash
{commands_str}