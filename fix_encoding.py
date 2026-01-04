#!/usr/bin/env python3
"""
Fix encoding issues for Windows
"""

import sys
import os

def fix_main_app():
    """Fix main_app.py encoding issues"""
    with open('main_app.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace Rupee symbol with simple text
    content = content.replace("₹", "Rs.")
    
    with open('main_app.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("Fixed main_app.py")

def fix_boq_generator():
    """Fix boq_generator.py encoding issues"""
    with open('boq_generator.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace Rupee symbol
    content = content.replace("₹", "Rs.")
    
    # Also fix the create_report method
    if "f.write(f'   Rate: ₹{row['Rate']:,.2f}')" in content:
        content = content.replace("f.write(f'   Rate: ₹{row['Rate']:,.2f}')", 
                                 "f.write(f'   Rate: {row['Rate']:,.2f}')")
    
    with open('boq_generator.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("Fixed boq_generator.py")

def main():
    print("Fixing encoding issues for Windows...")
    print("="*60)
    
    fix_main_app()
    fix_boq_generator()
    
    print("\n" + "="*60)
    print("✅ Fixes applied successfully!")
    print("\nNow run: python main_app.py")
    print("="*60)

if __name__ == "__main__":
    main()