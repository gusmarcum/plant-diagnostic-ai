# Knowledge Graph Data

This folder contains the FAOSTAT agricultural knowledge graph data used by the Plant Diagnostic System.

## Files

- **`kg_nodes_faostat.csv`** - Knowledge graph nodes containing agricultural entities and concepts
- **`kg_relationships_faostat.csv`** - Knowledge graph edges defining relationships between entities

## Usage

These files are used by the Plant Diagnostic System to provide:
- **Interactive Knowledge Graph**: Visual representation of agricultural data relationships
- **Crop-specific Insights**: Contextual information about crops and their characteristics
- **Agricultural Context**: Additional information to enhance diagnostic reports

## Data Source

The data is sourced from FAOSTAT (Food and Agriculture Organization of the United Nations) and provides comprehensive agricultural statistics and relationships.

## Integration

The knowledge graph is integrated into the demo interface and can be accessed through the "Knowledge Graph" tab in the web interface. Users can:
- Reload the full graph
- Show crop neighborhood relationships
- Explore agricultural data connections

## File Sizes

- `kg_nodes_faostat.csv`: ~14 KB (516 lines)
- `kg_relationships_faostat.csv`: ~1.6 MB (87,155 lines)

These files are essential for the enhanced analysis features of the Plant Diagnostic System.
