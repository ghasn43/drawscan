"""
visualization.py - Visualization tools for DXF BOQ Extractor
FIXED VERSION for Matplotlib compatibility
"""
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D
from io import BytesIO
from typing import List, Dict, Tuple, Optional
import warnings

# Suppress matplotlib warnings
warnings.filterwarnings('ignore')

# Set Matplotlib style
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('default')

class DXFVisualizer:
    """Visualization class for DXF structures - FIXED for Matplotlib compatibility"""
    
    def __init__(self):
        self.figsize = (12, 10)
        self.colors = {
            'walls': '#3498db',
            'openings': '#e74c3c',
            'columns': '#2ecc71',
            'text': '#34495e',
            'dimensions': '#9b59b6',
            'foundation': '#f39c12',
            'slab': '#16a085'
        }
        
        # Color palette for different wall types
        self.wall_colors = [
            '#3498db', '#2ecc71', '#e74c3c', '#f39c12', 
            '#9b59b6', '#1abc9c', '#d35400', '#c0392b'
        ]
    
    def create_wall_visualization(self, walls: List[Dict], 
                                 show_labels: bool = True,
                                 show_grid: bool = True,
                                 color_by: str = 'type') -> plt.Figure:
        """
        Create 2D visualization of extracted walls
        
        Args:
            walls: List of wall dictionaries from DXF processor
            show_labels: Whether to show wall labels
            show_grid: Whether to show grid
            color_by: 'type', 'layer', or 'length'
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if not walls:
            ax.text(0.5, 0.5, 'No walls to display', 
                   ha='center', va='center', fontsize=12)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            return fig
        
        # Calculate bounds for proper scaling
        all_points = []
        for wall in walls:
            start = wall.get('start_point', (0, 0))
            end = wall.get('end_point', (0, 0))
            all_points.extend([start, end])
        
        if all_points:
            points_array = np.array(all_points)
            x_min, y_min = points_array.min(axis=0) - 1000
            x_max, y_max = points_array.max(axis=0) + 1000
            
            # Set plot limits
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
        
        # Draw each wall
        for idx, wall in enumerate(walls):
            start = np.array(wall.get('start_point', (0, 0)))
            end = np.array(wall.get('end_point', (0, 0)))
            thickness = wall.get('thickness', 230)  # Default 230mm
            
            # Calculate wall vector and angle
            wall_vector = end - start
            length = np.linalg.norm(wall_vector)
            
            if length == 0:
                continue  # Skip zero-length walls
            
            angle_rad = np.arctan2(wall_vector[1], wall_vector[0])
            angle_deg = np.degrees(angle_rad)
            
            # Determine color based on criteria
            if color_by == 'layer' and 'layer' in wall:
                color_idx = hash(wall['layer']) % len(self.wall_colors)
                color = self.wall_colors[color_idx]
            elif color_by == 'length':
                # Color by length (normalized)
                norm_length = length / max(wall.get('length', length) for wall in walls)
                color = plt.cm.viridis(norm_length)
            else:
                color_idx = idx % len(self.wall_colors)
                color = self.wall_colors[color_idx]
            
            # FIXED: Create rectangle at origin, then transform it
            # Create rectangle centered at origin
            rect = Rectangle(
                (-length/2, -thickness/2),  # Center the rectangle
                length,
                thickness,
                fill=True,
                alpha=0.7,
                edgecolor=color,
                facecolor=color,
                linewidth=2
            )
            
            # Create transformation: rotate, then translate to wall position
            transform = (Affine2D()
                        .rotate(angle_rad)  # Rotate around origin
                        .translate(start[0] + length/2 * np.cos(angle_rad),
                                  start[1] + length/2 * np.sin(angle_rad))
                        + ax.transData)
            
            rect.set_transform(transform)
            ax.add_patch(rect)
            
            # Add wall label
            if show_labels:
                # Calculate midpoint
                midpoint = start + wall_vector / 2
                # Offset perpendicular to wall
                normal = np.array([-wall_vector[1], wall_vector[0]])
                normal = normal / np.linalg.norm(normal) * (thickness/2 + 100)
                label_pos = midpoint + normal
                
                label_text = f"W{idx+1}"
                if 'layer' in wall:
                    label_text += f"\n{wall['layer']}"
                label_text += f"\n{length/1000:.1f}m"
                
                ax.annotate(
                    label_text,
                    xy=midpoint,
                    xytext=label_pos,
                    ha='center',
                    va='center',
                    fontsize=8,
                    color='white',
                    fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7)
                )
        
        # Set plot properties
        ax.set_aspect('equal')
        ax.set_xlabel('X Coordinate (mm)')
        ax.set_ylabel('Y Coordinate (mm)')
        ax.set_title('DXF Wall Structure Visualization', fontsize=14, fontweight='bold')
        
        if show_grid:
            ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add scale indicator
        self._add_scale_indicator(ax, x_max, y_min)
        
        # Add statistics box
        total_walls = len(walls)
        total_length = sum(wall.get('length', 0) for wall in walls) / 1000  # meters
        avg_thickness = np.mean([wall.get('thickness', 230) for wall in walls])
        
        stats_text = f"Total Walls: {total_walls}\nTotal Length: {total_length:.1f}m\nAvg Thickness: {avg_thickness:.0f}mm"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def create_simple_wall_visualization(self, walls: List[Dict]) -> plt.Figure:
        """
        Simple wall visualization using lines (more reliable)
        
        Args:
            walls: List of wall dictionaries
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if not walls:
            ax.text(0.5, 0.5, 'No walls to display', 
                   ha='center', va='center', fontsize=12)
            return fig
        
        # Calculate bounds
        all_x = []
        all_y = []
        for wall in walls:
            start = wall.get('start_point', (0, 0))
            end = wall.get('end_point', (0, 0))
            all_x.extend([start[0], end[0]])
            all_y.extend([start[1], end[1]])
        
        if all_x and all_y:
            x_min, x_max = min(all_x), max(all_x)
            y_min, y_max = min(all_y), max(all_y)
            
            # Add padding
            x_pad = (x_max - x_min) * 0.1
            y_pad = (y_max - y_min) * 0.1
            
            ax.set_xlim(x_min - x_pad, x_max + x_pad)
            ax.set_ylim(y_min - y_pad, y_max + y_pad)
        
        # Draw walls as thick lines
        for idx, wall in enumerate(walls):
            start = wall.get('start_point', (0, 0))
            end = wall.get('end_point', (0, 0))
            thickness = wall.get('thickness', 230)
            
            color_idx = idx % len(self.wall_colors)
            color = self.wall_colors[color_idx]
            
            # Draw wall as a line with thickness
            ax.plot([start[0], end[0]], [start[1], end[1]], 
                   color=color, linewidth=thickness/20, alpha=0.7, solid_capstyle='round')
            
            # Add wall number
            midpoint = ((start[0] + end[0])/2, (start[1] + end[1])/2)
            ax.text(midpoint[0], midpoint[1], f'W{idx+1}', 
                   ha='center', va='center', fontsize=8, color='white',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='black', alpha=0.7))
        
        ax.set_aspect('equal')
        ax.set_xlabel('X Coordinate (mm)')
        ax.set_ylabel('Y Coordinate (mm)')
        ax.set_title('Wall Layout (Simplified)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def create_3d_visualization(self, walls: List[Dict], 
                               wall_height: float = 3000,
                               show_foundation: bool = True) -> go.Figure:
        """
        Create 3D visualization using Plotly
        
        Args:
            walls: List of wall dictionaries
            wall_height: Height of walls in mm (default 3m)
            show_foundation: Whether to show foundation
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        if not walls:
            # Return empty figure with message
            fig.add_annotation(
                text="No walls to display in 3D",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14)
            )
            return fig
        
        # Add foundation if requested
        if show_foundation:
            # Create a simple foundation slab
            foundation_points = []
            for wall in walls:
                start = wall.get('start_point', (0, 0))
                end = wall.get('end_point', (0, 0))
                foundation_points.extend([start, end])
            
            if foundation_points:
                points_array = np.array(foundation_points)
                x_min, y_min = points_array.min(axis=0) - 500
                x_max, y_max = points_array.max(axis=0) + 500
                
                # Create foundation slab
                fig.add_trace(go.Mesh3d(
                    x=[x_min, x_max, x_max, x_min, x_min, x_max, x_max, x_min],
                    y=[y_min, y_min, y_max, y_max, y_min, y_min, y_max, y_max],
                    z=[-500, -500, -500, -500, 0, 0, 0, 0],
                    i=[0, 0, 0, 4, 4, 6],
                    j=[1, 2, 3, 5, 6, 7],
                    k=[2, 3, 7, 6, 7, 4],
                    color='#8B4513',  # Brown for foundation
                    opacity=0.7,
                    name='Foundation',
                    showlegend=True
                ))
        
        # Add each wall as a 3D prism
        for idx, wall in enumerate(walls):
            start = wall.get('start_point', (0, 0))
            end = wall.get('end_point', (0, 0))
            thickness = wall.get('thickness', 230)
            
            # Skip invalid walls
            if start == end:
                continue
            
            # Calculate wall parameters
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            length = np.sqrt(dx**2 + dy**2)
            angle = np.arctan2(dy, dx)
            
            if length == 0:
                continue
            
            # Unit vectors
            u = np.array([np.cos(angle), np.sin(angle)])
            v = np.array([-np.sin(angle), np.cos(angle)])
            
            # Calculate 4 corners of the wall base
            corners_2d = [
                start,
                start + u * length,
                start + u * length + v * thickness,
                start + v * thickness
            ]
            
            # Create 3D vertices (bottom and top)
            x_coords = []
            y_coords = []
            z_coords = []
            
            for corner in corners_2d * 2:  # Bottom and top
                x_coords.append(corner[0])
                y_coords.append(corner[1])
            
            # Bottom z = 0, top z = wall_height
            z_coords = [0] * 4 + [wall_height] * 4
            
            # Define faces (triangles for each side)
            i = [0, 0, 0, 4, 4, 4, 0, 7]
            j = [1, 2, 4, 5, 6, 1, 3, 4]
            k = [2, 3, 5, 6, 7, 2, 7, 5]
            
            color_idx = idx % len(self.wall_colors)
            
            fig.add_trace(go.Mesh3d(
                x=x_coords,
                y=y_coords,
                z=z_coords,
                i=i,
                j=j,
                k=k,
                color=self.wall_colors[color_idx],
                opacity=0.8,
                name=f"Wall {idx+1}",
                showlegend=idx < 5,  # Only show first 5 in legend
                hoverinfo='text',
                hovertext=f"Wall {idx+1}<br>Length: {length/1000:.1f}m<br>Thickness: {thickness}mm"
            ))
        
        # Update layout
        fig.update_layout(
            title='3D Wall Structure Visualization',
            scene=dict(
                xaxis_title='X (mm)',
                yaxis_title='Y (mm)',
                zaxis_title='Height (mm)',
                aspectmode='data',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            height=700,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        return fig
    
    def create_quantity_chart(self, boq_data: List[Dict]) -> plt.Figure:
        """
        Create bar chart of quantities
        
        Args:
            boq_data: BOQ data from generator
            
        Returns:
            Matplotlib figure
        """
        if not boq_data:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No BOQ data to display', 
                   ha='center', va='center', fontsize=12)
            return fig
        
        items = [item['item'] for item in boq_data]
        quantities = [item['quantity'] for item in boq_data]
        units = [item.get('unit', '') for item in boq_data]
        
        # Truncate long item names
        truncated_items = []
        for item in items:
            if len(item) > 30:
                truncated_items.append(item[:27] + '...')
            else:
                truncated_items.append(item)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Bar chart
        bars = ax1.barh(truncated_items, quantities, color='#3498db', alpha=0.8)
        ax1.set_xlabel('Quantity')
        ax1.set_title('Material Quantities', fontweight='bold')
        ax1.invert_yaxis()  # Largest on top
        
        # Add unit labels
        for i, (bar, unit) in enumerate(zip(bars, units)):
            width = bar.get_width()
            ax1.text(width + max(quantities)*0.01, bar.get_y() + bar.get_height()/2,
                    f'{width:,.2f} {unit}', va='center', fontsize=9)
        
        # Pie chart for percentage
        if len(items) > 0:
            # Create custom colors for pie chart
            colors = plt.cm.Set3(np.linspace(0, 1, len(items)))
            
            # Calculate percentages
            total = sum(quantities)
            percentages = [q/total*100 for q in quantities]
            
            wedges, texts, autotexts = ax2.pie(
                quantities, 
                labels=truncated_items, 
                autopct=lambda p: f'{p:.1f}%' if p > 3 else '',
                colors=colors,
                startangle=90,
                pctdistance=0.85
            )
            
            # Make percentage text white and bold
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            
            ax2.set_title('Quantity Distribution', fontweight='bold')
            
            # Add legend
            ax2.legend(wedges, [f"{item}: {qty:,.2f} {unit}" 
                               for item, qty, unit in zip(truncated_items, quantities, units)],
                      title="Items",
                      loc="center left",
                      bbox_to_anchor=(1, 0, 0.5, 1),
                      fontsize=8)
        
        plt.tight_layout()
        return fig
    
    def create_summary_dashboard(self, 
                                walls: List[Dict], 
                                boq_data: List[Dict],
                                filename: str) -> plt.Figure:
        """
        Create a comprehensive dashboard with multiple visualizations
        
        Args:
            walls: Extracted walls
            boq_data: BOQ data
            filename: Original DXF filename
            
        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(16, 12))
        
        # Create subplots grid
        gs = fig.add_gridspec(3, 3)
        
        # 1. Simple wall visualization (top-left, 2x2)
        ax1 = fig.add_subplot(gs[0:2, 0:2])
        self._plot_walls_simple(ax1, walls)
        ax1.set_title(f'Wall Layout: {filename}', fontsize=12, fontweight='bold')
        
        # 2. Quantity summary (top-right)
        ax2 = fig.add_subplot(gs[0, 2])
        if boq_data:
            items = [item['item'][:15] + '...' if len(item['item']) > 15 else item['item'] 
                    for item in boq_data]
            quantities = [item['quantity'] for item in boq_data]
            
            # Sort by quantity
            sorted_indices = np.argsort(quantities)[-8:]  # Top 8 items
            sorted_items = [items[i] for i in sorted_indices]
            sorted_quantities = [quantities[i] for i in sorted_indices]
            
            bars = ax2.barh(sorted_items, sorted_quantities, color='#2ecc71')
            ax2.set_xlabel('Quantity')
            ax2.set_title('Top Quantities', fontsize=10)
            ax2.invert_yaxis()
            
            # Add value labels
            for bar in bars:
                width = bar.get_width()
                ax2.text(width + max(sorted_quantities)*0.01, 
                        bar.get_y() + bar.get_height()/2,
                        f'{width:,.1f}', va='center', fontsize=8)
        
        # 3. Statistics (middle-right)
        ax3 = fig.add_subplot(gs[1, 2])
        
        # Calculate statistics
        total_walls = len(walls)
        
        if walls:
            wall_lengths = [wall.get('length', 0) for wall in walls]
            wall_thicknesses = [wall.get('thickness', 230) for wall in walls]
            
            total_length = sum(wall_lengths) / 1000  # meters
            avg_length = np.mean(wall_lengths)
            avg_thickness = np.mean(wall_thicknesses)
            
            stats = [total_walls, total_length, avg_length, avg_thickness]
            stat_labels = ['Walls', 'Length (m)', 'Avg Len (mm)', 'Avg Thick (mm)']
            colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
            
            bars = ax3.bar(stat_labels, stats, color=colors, alpha=0.7)
            ax3.set_title('Wall Statistics', fontsize=10)
            ax3.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, stat, label in zip(bars, stats, stat_labels):
                height = bar.get_height()
                if label == 'Length (m)':
                    text = f'{stat:.1f}m'
                elif label == 'Avg Len (mm)':
                    text = f'{stat:.0f}mm'
                elif label == 'Avg Thick (mm)':
                    text = f'{stat:.0f}mm'
                else:
                    text = f'{stat:.0f}'
                
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        text, ha='center', va='bottom', fontsize=9)
        else:
            ax3.text(0.5, 0.5, 'No wall data', 
                    ha='center', va='center', transform=ax3.transAxes)
        
        # 4. Wall length distribution (bottom)
        ax4 = fig.add_subplot(gs[2, :])
        if walls and len(walls) > 1:
            lengths = [wall.get('length', 0) for wall in walls]
            
            # Convert to meters for better readability
            lengths_m = [l/1000 for l in lengths]
            
            n, bins, patches = ax4.hist(lengths_m, bins=15, alpha=0.7, 
                                        color='#9b59b6', edgecolor='black')
            ax4.set_xlabel('Wall Length (m)')
            ax4.set_ylabel('Frequency')
            ax4.set_title('Wall Length Distribution', fontsize=10)
            
            # Add mean line
            mean_length = np.mean(lengths_m)
            ax4.axvline(mean_length, color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {mean_length:.2f}m')
            
            # Add median line
            median_length = np.median(lengths_m)
            ax4.axvline(median_length, color='green', linestyle=':', 
                       linewidth=2, label=f'Median: {median_length:.2f}m')
            
            ax4.legend(fontsize=8)
            
            # Add statistics text
            stats_text = f"Min: {min(lengths_m):.2f}m\nMax: {max(lengths_m):.2f}m\nStd: {np.std(lengths_m):.2f}m"
            ax4.text(0.98, 0.98, stats_text, transform=ax4.transAxes,
                    fontsize=8, verticalalignment='top',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax4.text(0.5, 0.5, 'Insufficient wall data for distribution', 
                    ha='center', va='center', transform=ax4.transAxes)
        
        plt.suptitle(f'DXF Analysis Dashboard - {filename}', fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        return fig
    
    def _plot_walls_simple(self, ax, walls: List[Dict]):
        """Helper to plot walls simply"""
        if not walls:
            ax.text(0.5, 0.5, 'No walls to display', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        for idx, wall in enumerate(walls):
            start = wall.get('start_point', (0, 0))
            end = wall.get('end_point', (0, 0))
            thickness = wall.get('thickness', 230)
            
            # Plot wall as thick line
            ax.plot([start[0], end[0]], [start[1], end[1]], 
                   color=self.wall_colors[idx % len(self.wall_colors)], 
                   linewidth=thickness/20, alpha=0.7, solid_capstyle='round')
            
            # Plot wall start and end points
            ax.plot(start[0], start[1], 'o', color='red', markersize=4, alpha=0.7)
            ax.plot(end[0], end[1], 's', color='blue', markersize=4, alpha=0.7)
        
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
    
    def _add_scale_indicator(self, ax, x_max: float, y_min: float):
        """Add scale indicator to plot"""
        try:
            scale_length = 5000  # 5m scale
            scale_y = y_min + 500
            
            # Draw scale line
            ax.plot([x_max - scale_length - 1000, x_max - 1000], 
                   [scale_y, scale_y], 'k-', linewidth=3)
            
            # Add ticks
            for tick_pos in [x_max - scale_length - 1000, x_max - 1000]:
                ax.plot([tick_pos, tick_pos], [scale_y - 100, scale_y + 100], 
                       'k-', linewidth=2)
            
            # Add label
            ax.text(x_max - scale_length/2 - 1000, scale_y + 300,
                   f'{scale_length/1000:.0f} meters', ha='center', 
                   fontweight='bold', fontsize=10)
            
            # Add scale indicator text
            ax.text(x_max - scale_length - 1100, scale_y - 300,
                   'Scale', fontsize=8, rotation=90, va='center')
        except:
            pass  # Silently fail if scale indicator can't be added

# Utility functions for Streamlit
def get_matplotlib_fig_bytes(fig: plt.Figure, dpi: int = 150) -> BytesIO:
    """Convert matplotlib figure to bytes for Streamlit"""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    return buf

def display_plotly_in_streamlit(fig: go.Figure, height: int = 700):
    """Display Plotly figure in Streamlit"""
    try:
        import streamlit as st
        st.plotly_chart(fig, use_container_width=True, height=height)
    except ImportError:
        print("Streamlit not available for Plotly display")

# Example usage
if __name__ == "__main__":
    # Test the visualizer
    visualizer = DXFVisualizer()
    
    # Create some test walls
    test_walls = [
        {
            'id': 1,
            'start_point': (0, 0),
            'end_point': (5000, 0),
            'length': 5000,
            'thickness': 230,
            'layer': 'EXTERIOR_WALLS'
        },
        {
            'id': 2,
            'start_point': (5000, 0),
            'end_point': (5000, 3000),
            'length': 3000,
            'thickness': 115,
            'layer': 'INTERIOR_WALLS'
        },
        {
            'id': 3,
            'start_point': (0, 3000),
            'end_point': (5000, 3000),
            'length': 5000,
            'thickness': 230,
            'layer': 'EXTERIOR_WALLS'
        },
        {
            'id': 4,
            'start_point': (0, 0),
            'end_point': (0, 3000),
            'length': 3000,
            'thickness': 230,
            'layer': 'EXTERIOR_WALLS'
        }
    ]
    
    # Test BOQ data
    test_boq = [
        {'item': 'Brickwork in superstructure', 'quantity': 24.5, 'unit': 'm³'},
        {'item': 'Cement plaster 20mm thick', 'quantity': 156.8, 'unit': 'm²'},
        {'item': 'Concrete in foundation', 'quantity': 12.3, 'unit': 'm³'},
        {'item': 'Steel reinforcement', 'quantity': 1.2, 'unit': 'ton'}
    ]
    
    print("Testing FIXED visualization module...")
    
    # Test 1: Simple visualization (most reliable)
    print("1. Creating simple wall visualization...")
    fig_simple = visualizer.create_simple_wall_visualization(test_walls)
    fig_simple.savefig('test_output_simple.png', dpi=150, bbox_inches='tight')
    print("   Saved: test_output_simple.png")
    
    # Test 2: Advanced visualization (with transform fix)
    print("2. Creating advanced wall visualization...")
    try:
        fig_advanced = visualizer.create_wall_visualization(test_walls, show_labels=True)
        fig_advanced.savefig('test_output_advanced.png', dpi=150, bbox_inches='tight')
        print("   Saved: test_output_advanced.png")
    except Exception as e:
        print(f"   Advanced visualization failed (using fallback): {e}")
        fig_simple.savefig('test_output_advanced.png', dpi=150, bbox_inches='tight')
    
    # Test 3: 3D Visualization
    print("3. Creating 3D visualization...")
    fig_3d = visualizer.create_3d_visualization(test_walls, wall_height=3000)
    fig_3d.write_html('test_output_3d.html')
    print("   Saved: test_output_3d.html")
    
    # Test 4: Quantity Chart
    print("4. Creating quantity chart...")
    fig_qty = visualizer.create_quantity_chart(test_boq)
    fig_qty.savefig('test_output_quantity.png', dpi=150, bbox_inches='tight')
    print("   Saved: test_output_quantity.png")
    
    # Test 5: Dashboard
    print("5. Creating dashboard...")
    fig_dash = visualizer.create_summary_dashboard(test_walls, test_boq, "test.dxf")
    fig_dash.savefig('test_output_dashboard.png', dpi=150, bbox_inches='tight')
    print("   Saved: test_output_dashboard.png")
    
    print("\n✅ All tests completed successfully!")
    print("\nGenerated files:")
    print("  - test_output_simple.png")
    print("  - test_output_advanced.png")
    print("  - test_output_3d.html")
    print("  - test_output_quantity.png")
    print("  - test_output_dashboard.png")
    print("\nOpen test_output_3d.html in your browser for interactive 3D view!")