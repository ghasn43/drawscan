#!/usr/bin/env python3
"""
Core DXF Processing Engine
Handles DXF file reading, geometry extraction, and unit conversion
"""

import ezdxf
import math
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass
class DXFEntity:
    """Data class for DXF entity information"""
    entity_type: str
    layer: str
    length_m: float = 0.0
    area_m2: float = 0.0
    vertices: List[Tuple[float, float]] = None
    properties: Dict[str, Any] = None

    def __post_init__(self):
        if self.vertices is None:
            self.vertices = []
        if self.properties is None:
            self.properties = {}


@dataclass
class LayerInfo:
    """Data class for layer information"""
    name: str
    color: int
    entity_count: int
    entity_types: Dict[str, int]
    suggested_purpose: str = ""


class DXFProcessor:
    """Main DXF processing class"""

    # Layer name patterns for automatic classification
    LAYER_PATTERNS = {
        'walls': ['WALL', 'WALLS', 'STRUCT_WALL', 'A-WALL'],
        'doors': ['DOOR', 'DOORS', 'A-DOOR', 'DR'],
        'windows': ['WINDOW', 'WINDOWS', 'A-WIND', 'WD', 'WIND'],
        'slabs': ['SLAB', 'SLABS', 'FLOOR', 'A-SLAB'],
        'beams': ['BEAM', 'BEAMS', 'A-BEAM'],
        'columns': ['COLUMN', 'COLUMNS', 'A-COLUMN'],
        'text': ['TEXT', 'TEXTS', 'ANNOTATION', 'A-ANNO', 'NOTES'],
        'dimensions': ['DIM', 'DIMS', 'DIMENSION', 'A-DIMS'],
        'furniture': ['FURNITURE', 'FURN', 'A-FURN'],
        'plumbing': ['PLUMBING', 'PIPE', 'A-PLUMB'],
        'electrical': ['ELECTRICAL', 'ELEC', 'A-ELEC'],
        'grid': ['GRID', 'GRIDS', 'AXIS'],
    }

    # Standard dimensions (in meters)
    STANDARDS = {
        'wall_thickness': 0.23,      # 230mm
        'slab_thickness': 0.15,      # 150mm
        'door_width': 0.90,          # 900mm
        'door_height': 2.10,         # 2100mm
        'window_height': 1.20,       # 1200mm
        'beam_width': 0.23,          # 230mm
        'beam_depth': 0.45,          # 450mm
        'column_size': 0.23,         # 230mm square
    }

    def __init__(self, units: str = 'm'):
        """
        Initialize DXF processor

        Args:
            units: Drawing units ('mm', 'cm', 'm')
        """
        self.units = units.lower()
        self.conversion_factor = self._get_conversion_factor()
        self.doc = None
        self.msp = None

    def _get_conversion_factor(self) -> float:
        """Get conversion factor from drawing units to meters"""
        factors = {
            'mm': 0.001,
            'cm': 0.01,
            'm': 1.0,
            'inch': 0.0254,
            'ft': 0.3048,
        }
        return factors.get(self.units, 1.0)

    def load_file(self, filepath: Path) -> bool:
        """Load a DXF file"""
        try:
            self.doc = ezdxf.readfile(str(filepath))
            self.msp = self.doc.modelspace()
            return True
        except Exception as e:
            print(f"Error loading DXF: {e}")
            return False

    def get_drawing_info(self) -> Dict:
        """Get basic drawing information"""
        if not self.doc:
            return {}

        info = {
            'units': self.units,
            'layers': [],
            'entity_count': len(list(self.msp)),
            'layer_count': len(list(self.doc.layers)),
            'extents': self._get_drawing_extents()
        }

        return info

    def _get_drawing_extents(self) -> Dict:
        """Get drawing bounding box"""
        if not self.msp:
            return {}

        all_points = []
        for entity in self.msp:
            try:
                extents = entity.get_extents()
                if extents:
                    all_points.extend(extents)
            except:
                continue

        if not all_points:
            return {}

        xs = [p[0] for p in all_points]
        ys = [p[1] for p in all_points]

        return {
            'min_x': min(xs),
            'min_y': min(ys),
            'max_x': max(xs),
            'max_y': max(ys),
            'width': max(xs) - min(xs),
            'height': max(ys) - min(ys)
        }

    def analyze_layers(self) -> List[LayerInfo]:
        """Analyze all layers in the drawing"""
        if not self.doc:
            return []

        layers = []
        for layer in self.doc.layers:
            layer_name = layer.dxf.name
            entities = list(self.msp.query(f'*[layer=="{layer_name}"]'))

            # Count entity types
            entity_types = {}
            for entity in entities:
                etype = entity.dxftype()
                entity_types[etype] = entity_types.get(etype, 0) + 1

            # Suggest purpose based on layer name
            suggested_purpose = self._suggest_layer_purpose(layer_name)

            layers.append(LayerInfo(
                name=layer_name,
                color=layer.dxf.color,
                entity_count=len(entities),
                entity_types=entity_types,
                suggested_purpose=suggested_purpose
            ))

        return layers

    def _suggest_layer_purpose(self, layer_name: str) -> str:
        """Suggest layer purpose based on name patterns"""
        layer_upper = layer_name.upper()

        for purpose, patterns in self.LAYER_PATTERNS.items():
            for pattern in patterns:
                if pattern in layer_upper:
                    return purpose

        return "unknown"

    def extract_entities(self) -> List[DXFEntity]:
        """Extract all entities with geometry data"""
        if not self.msp:
            return []

        entities = []
        for entity in self.msp:
            dxf_entity = self._extract_entity_data(entity)
            if dxf_entity:
                entities.append(dxf_entity)

        return entities

    def _extract_entity_data(self, entity) -> Optional[DXFEntity]:
        """Extract data from a single entity"""
        entity_type = entity.dxftype()
        layer = entity.dxf.layer

        try:
            if entity_type == 'LINE':
                return self._extract_line(entity, layer)
            elif entity_type in ['LWPOLYLINE', 'POLYLINE']:
                return self._extract_polyline(entity, layer)
            elif entity_type == 'CIRCLE':
                return self._extract_circle(entity, layer)
            elif entity_type == 'ARC':
                return self._extract_arc(entity, layer)
            elif entity_type in ['TEXT', 'MTEXT']:
                return self._extract_text(entity, layer)
            elif entity_type == 'INSERT':
                return self._extract_insert(entity, layer)
            elif entity_type == 'DIMENSION':
                return self._extract_dimension(entity, layer)
            else:
                return None

        except Exception as e:
            print(f"Warning: Could not extract {entity_type}: {e}")
            return None

    def _extract_line(self, entity, layer: str) -> DXFEntity:
        """Extract line data"""
        start = entity.dxf.start
        end = entity.dxf.end

        # Get coordinates
        if hasattr(start, 'x'):
            x1, y1 = start.x, start.y
            x2, y2 = end.x, end.y
        else:
            x1, y1 = float(start[0]), float(start[1])
            x2, y2 = float(end[0]), float(end[1])

        # Calculate length
        length_units = math.sqrt((x2-x1)**2 + (y2-y1)**2)
        length_m = length_units * self.conversion_factor

        # Classify based on layer
        element_type = self._classify_element(layer, 'LINEAR')

        return DXFEntity(
            entity_type='LINE',
            layer=layer,
            length_m=length_m,
            vertices=[(x1, y1), (x2, y2)],
            properties={
                'element_type': element_type,
                'start': (x1, y1),
                'end': (x2, y2),
                'length_units': length_units
            }
        )

    def _extract_polyline(self, entity, layer: str) -> DXFEntity:
        """Extract polyline data"""
        vertices = []

        # Get vertices
        try:
            if hasattr(entity, 'get_points'):
                points = entity.get_points()
                for point in points:
                    vertices.append((float(point[0]), float(point[1])))
            elif hasattr(entity, 'vertices'):
                for vtx in entity.vertices():
                    vertices.append((float(vtx[0]), float(vtx[1])))
        except:
            return None

        if len(vertices) < 2:
            return None

        # Calculate length
        length_units = 0
        for i in range(len(vertices)-1):
            x1, y1 = vertices[i]
            x2, y2 = vertices[i+1]
            length_units += math.sqrt((x2-x1)**2 + (y2-y1)**2)

        # Add closing segment if closed
        if hasattr(entity, 'closed') and entity.closed and len(vertices) > 1:
            x1, y1 = vertices[-1]
            x2, y2 = vertices[0]
            length_units += math.sqrt((x2-x1)**2 + (y2-y1)**2)

        length_m = length_units * self.conversion_factor

        # Calculate area if closed polyline
        area_m2 = 0
        if hasattr(entity, 'closed') and entity.closed and len(vertices) >= 3:
            area_units = self._calculate_polygon_area(vertices)
            area_m2 = area_units * (self.conversion_factor ** 2)

        # Classify based on layer
        element_type = self._classify_element(layer, 'AREA' if entity.closed else 'LINEAR')

        return DXFEntity(
            entity_type='POLYLINE',
            layer=layer,
            length_m=length_m,
            area_m2=area_m2,
            vertices=vertices,
            properties={
                'element_type': element_type,
                'closed': getattr(entity, 'closed', False),
                'vertex_count': len(vertices),
                'length_units': length_units,
                'area_units': area_units if area_m2 > 0 else 0
            }
        )

    def _extract_circle(self, entity, layer: str) -> DXFEntity:
        """Extract circle data"""
        radius_units = float(entity.dxf.radius)
        radius_m = radius_units * self.conversion_factor

        # Calculate area
        area_m2 = math.pi * radius_m * radius_m

        # Classify based on layer
        element_type = self._classify_element(layer, 'AREA')

        # Generate points for circle
        vertices = []
        for i in range(36):  # 36 points for smooth circle
            angle = 2 * math.pi * i / 36
            x = entity.dxf.center.x + radius_units * math.cos(angle)
            y = entity.dxf.center.y + radius_units * math.sin(angle)
            vertices.append((float(x), float(y)))

        return DXFEntity(
            entity_type='CIRCLE',
            layer=layer,
            area_m2=area_m2,
            vertices=vertices,
            properties={
                'element_type': element_type,
                'center': (entity.dxf.center.x, entity.dxf.center.y),
                'radius_units': radius_units,
                'radius_m': radius_m,
                'diameter_m': radius_m * 2
            }
        )

    def _extract_arc(self, entity, layer: str) -> DXFEntity:
        """Extract arc data"""
        radius_units = float(entity.dxf.radius)
        radius_m = radius_units * self.conversion_factor
        start_angle = math.radians(entity.dxf.start_angle)
        end_angle = math.radians(entity.dxf.end_angle)

        # Calculate arc length
        angle_diff = end_angle - start_angle
        if angle_diff < 0:
            angle_diff += 2 * math.pi

        length_m = radius_m * angle_diff

        return DXFEntity(
            entity_type='ARC',
            layer=layer,
            length_m=length_m,
            properties={
                'element_type': 'ARC',
                'center': (entity.dxf.center.x, entity.dxf.center.y),
                'radius_units': radius_units,
                'radius_m': radius_m,
                'start_angle': entity.dxf.start_angle,
                'end_angle': entity.dxf.end_angle
            }
        )

    def _extract_text(self, entity, layer: str) -> DXFEntity:
        """Extract text data"""
        if entity.dxftype() == 'TEXT':
            text_content = entity.dxf.text
        else:  # MTEXT
            text_content = entity.text

        text_content = str(text_content).strip()

        return DXFEntity(
            entity_type='TEXT',
            layer=layer,
            properties={
                'element_type': 'TEXT',
                'content': text_content,
                'height': entity.dxf.height * self.conversion_factor,
                'position': (entity.dxf.insert.x, entity.dxf.insert.y),
                'rotation': entity.dxf.rotation
            }
        )

    def _extract_insert(self, entity, layer: str) -> DXFEntity:
        """Extract block insert data"""
        block_name = entity.dxf.name

        # Determine block type
        block_type = self._classify_block(block_name, layer)

        return DXFEntity(
            entity_type='INSERT',
            layer=layer,
            properties={
                'element_type': block_type,
                'block_name': block_name,
                'position': (entity.dxf.insert.x, entity.dxf.insert.y),
                'rotation': entity.dxf.rotation,
                'scale': (entity.dxf.xscale, entity.dxf.yscale)
            }
        )

    def _extract_dimension(self, entity, layer: str) -> DXFEntity:
        """Extract dimension data"""
        # Try to get dimension value
        measurement = None
        if hasattr(entity.dxf, 'text'):
            import re
            numbers = re.findall(r'\d+\.?\d*', str(entity.dxf.text))
            if numbers:
                measurement_units = float(numbers[0])
                measurement = measurement_units * self.conversion_factor

        return DXFEntity(
            entity_type='DIMENSION',
            layer=layer,
            properties={
                'element_type': 'DIMENSION',
                'measurement_m': measurement,
                'text': str(getattr(entity.dxf, 'text', '')),
                'dimstyle': entity.dxf.dimstyle
            }
        )

    def _classify_element(self, layer: str, category: str) -> str:
        """Classify element based on layer name"""
        purpose = self._suggest_layer_purpose(layer)

        if purpose != 'unknown':
            return purpose.upper()

        # Default classification
        if category == 'LINEAR':
            return 'LINEAR_ELEMENT'
        elif category == 'AREA':
            return 'AREA_ELEMENT'
        else:
            return 'OTHER'

    def _classify_block(self, block_name: str, layer: str) -> str:
        """Classify block type"""
        block_upper = block_name.upper()
        layer_upper = layer.upper()

        # Check block name
        if any(keyword in block_upper for keyword in ['DOOR', 'DR']):
            return 'DOOR'
        elif any(keyword in block_upper for keyword in ['WINDOW', 'WD', 'WIND']):
            return 'WINDOW'
        elif any(keyword in block_upper for keyword in ['COLUMN', 'COL']):
            return 'COLUMN'
        elif any(keyword in block_upper for keyword in ['TOILET', 'WC', 'SINK']):
            return 'FIXTURE'

        # Check layer name
        if any(keyword in layer_upper for keyword in ['DOOR', 'DR']):
            return 'DOOR'
        elif any(keyword in layer_upper for keyword in ['WINDOW', 'WD', 'WIND']):
            return 'WINDOW'

        return 'BLOCK'

    def _calculate_polygon_area(self, vertices: List[Tuple[float, float]]) -> float:
        """Calculate polygon area using shoelace formula"""
        n = len(vertices)
        if n < 3:
            return 0.0

        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += vertices[i][0] * vertices[j][1]
            area -= vertices[j][0] * vertices[i][1]

        return abs(area) / 2.0

    def calculate_quantities(self, entities: List[DXFEntity]) -> Dict:
        """Calculate quantities for BOQ"""
        quantities = {
            'walls': {'length_m': 0, 'area_m2': 0, 'count': 0},
            'beams': {'length_m': 0, 'count': 0},
            'columns': {'count': 0, 'volume_m3': 0},
            'slabs': {'area_m2': 0, 'volume_m3': 0, 'count': 0},
            'doors': {'count': 0},
            'windows': {'count': 0},
            'linear_elements': {'length_m': 0, 'count': 0},
            'area_elements': {'area_m2': 0, 'count': 0},
            'text_annotations': {'count': 0},
        }

        for entity in entities:
            element_type = entity.properties.get('element_type', '').lower()

            if 'wall' in element_type:
                quantities['walls']['length_m'] += entity.length_m
                quantities['walls']['count'] += 1

                # Calculate wall area (length × standard thickness)
                if entity.length_m > 0:
                    wall_area = entity.length_m * self.STANDARDS['wall_thickness']
                    quantities['walls']['area_m2'] += wall_area

            elif 'beam' in element_type:
                quantities['beams']['length_m'] += entity.length_m
                quantities['beams']['count'] += 1

            elif 'column' in element_type:
                quantities['columns']['count'] += 1
                # Calculate column volume (standard size)
                column_volume = (self.STANDARDS['column_size'] ** 2) * 3.0  # Assuming 3m height
                quantities['columns']['volume_m3'] += column_volume

            elif 'slab' in element_type or 'floor' in element_type:
                quantities['slabs']['area_m2'] += entity.area_m2
                quantities['slabs']['count'] += 1

                # Calculate slab volume (area × standard thickness)
                if entity.area_m2 > 0:
                    slab_volume = entity.area_m2 * self.STANDARDS['slab_thickness']
                    quantities['slabs']['volume_m3'] += slab_volume

            elif 'door' in element_type:
                quantities['doors']['count'] += 1

            elif 'window' in element_type:
                quantities['windows']['count'] += 1

            elif 'linear' in element_type and entity.length_m > 0:
                quantities['linear_elements']['length_m'] += entity.length_m
                quantities['linear_elements']['count'] += 1

            elif 'area' in element_type and entity.area_m2 > 0:
                quantities['area_elements']['area_m2'] += entity.area_m2
                quantities['area_elements']['count'] += 1

            elif entity.entity_type in ['TEXT', 'MTEXT']:
                quantities['text_annotations']['count'] += 1

        return quantities

    def process_dxf(self, file_path: str, default_thickness: float = 230.0) -> List[Dict]:
        """
        Process DXF file and extract walls in standard format
        Compatibility method for streamlit_app.py
        
        Args:
            file_path: Path to DXF file
            default_thickness: Default wall thickness in mm
            
        Returns:
            List of wall dictionaries with standardized format
        """
        try:
            # Load the DXF file
            success = self.load_file(Path(file_path))
            if not success:
                print(f"Failed to load DXF file: {file_path}")
                return []
            
            # Extract entities
            dxf_entities = self.extract_entities()
            
            # Convert DXFEntity objects to wall dictionaries
            walls = []
            
            for idx, entity in enumerate(dxf_entities):
                # Only process line-like entities (walls)
                if entity.entity_type in ['LINE', 'LWPOLYLINE', 'POLYLINE']:
                    # Get vertices
                    vertices = entity.vertices
                    if len(vertices) >= 2:
                        # For polylines with multiple segments, create walls for each segment
                        for i in range(len(vertices) - 1):
                            start = vertices[i]
                            end = vertices[i + 1]
                            
                            # Calculate length
                            dx = end[0] - start[0]
                            dy = end[1] - start[1]
                            length_units = math.sqrt(dx*dx + dy*dy)
                            
                            # Convert to meters then to mm for consistency with streamlit_app
                            length_mm = length_units * self.conversion_factor * 1000
                            
                            # Skip very short walls (less than 100mm)
                            if length_mm < 100:
                                continue
                            
                            # Create wall dictionary
                            wall = {
                                'id': len(walls) + 1,
                                'type': entity.entity_type,
                                'start_point': start,
                                'end_point': end,
                                'length': length_mm,  # in mm for compatibility
                                'thickness': default_thickness,
                                'layer': entity.layer,
                                'color': 7,  # Default color
                                'angle': math.degrees(math.atan2(dy, dx)) if length_units > 0 else 0,
                                'midpoint': (
                                    (start[0] + end[0]) / 2,
                                    (start[1] + end[1]) / 2
                                ),
                                'original_data': {
                                    'entity_type': entity.entity_type,
                                    'properties': entity.properties
                                }
                            }
                            
                            walls.append(wall)
            
            return walls
            
        except Exception as e:
            print(f"Error processing DXF file {file_path}: {e}")
            import traceback
            traceback.print_exc()
            return []


# Utility function for quick processing
def process_dxf_file(file_path: str, default_thickness: float = 230.0) -> List[Dict]:
    """
    Quick utility function to process DXF file
    
    Args:
        file_path: Path to DXF file
        default_thickness: Default wall thickness in mm
        
    Returns:
        List of wall dictionaries
    """
    processor = DXFProcessor()
    return processor.process_dxf(file_path, default_thickness)


if __name__ == "__main__":
    # Test the processor
    import sys
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        try:
            processor = DXFProcessor()
            walls = processor.process_dxf(file_path)
            
            print(f"✅ Processed: {file_path}")
            print(f"📊 Statistics:")
            print(f"  Total walls: {len(walls)}")
            
            if walls:
                total_length_mm = sum(wall['length'] for wall in walls)
                total_length_m = total_length_mm / 1000
                print(f"  Total length: {total_length_m:.2f} m ({total_length_mm:.0f} mm)")
                
                # Print first few walls
                print(f"\n🏗️ First 5 walls:")
                for i, wall in enumerate(walls[:5]):
                    print(f"  Wall {i+1}: {wall['length']:.0f}mm on layer '{wall['layer']}'")
                    
                # Test extract_entities
                entities = processor.extract_entities()
                print(f"\n📈 Entity Statistics:")
                print(f"  Total entities: {len(entities)}")
                
                # Count by type
                entity_counts = {}
                for entity in entities:
                    entity_type = entity.entity_type
                    entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
                
                for etype, count in entity_counts.items():
                    print(f"  {etype}: {count}")
                    
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Usage: python dxf_processor.py <dxf_file_path>")
        print("Example: python dxf_processor.py test_building.dxf")