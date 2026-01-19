"""
Visualization Utilities.
File: src/rta/utils/visualization.py
"""

def generate_reasoning_graph(result, output_format: str = "mermaid") -> str:
    """
    Generates a Mermaid JS graph definition from the reasoning result.
    """
    lines = ["graph TD"]
    
    # Root Node
    # Use getattr to safely retrieve the topic or default to "Research Topic"
    topic = getattr(result, 'topic', "Research Topic")
    # Sanitize string to prevent syntax errors in Mermaid
    safe_topic = topic.replace('"', '').replace("'", "")
    lines.append(f"    Root([\"{safe_topic}\"])")
    
    # Styles
    lines.append("    classDef cluster fill:#e1f5fe,stroke:#01579b,stroke-width:2px;")
    lines.append("    classDef finding fill:#fff9c4,stroke:#fbc02d,stroke-width:1px;")

    # Iterate through clusters
    clusters = getattr(result, 'clusters', [])
    
    for i, cluster in enumerate(clusters):
        # [FIXED] Changed 'topic_name' to 'name' to match the schema
        raw_name = getattr(cluster, 'name', getattr(cluster, 'topic_name', f"Cluster_{i}"))
        safe_cluster_name = raw_name.replace('"', '')
        
        cluster_node_id = f"C{i}"
        lines.append(f"    Root --> {cluster_node_id}[\"{safe_cluster_name}\"]")
        lines.append(f"    class {cluster_node_id} cluster")
        
        # Iterate through findings (if available)
        findings = getattr(cluster, 'key_findings', [])
        for j, finding in enumerate(findings):
            summary = getattr(finding, 'summary', 'Finding')
            # Truncate long summaries for visual clarity
            short_summary = (summary[:40] + "...") if len(summary) > 40 else summary
            safe_summary = short_summary.replace('"', '')
            
            finding_node_id = f"F{i}_{j}"
            lines.append(f"    {cluster_node_id} --> {finding_node_id}(\"{safe_summary}\")")
            lines.append(f"    class {finding_node_id} finding")

    return "\n".join(lines)