def get_subtasks(task_name):
    """
    Get the subtasks for a given task name.
    """
    subtasks = ['Art_Style', 'Functional_Correspondence', 'Multi-view_Reasoning', 'Relative_Reflectance', 'Visual_Correspondence', 'Counting', 'IQ_Test', 'Object_Localization', 'Semantic_Correspondence', 'Visual_Similarity', 'Forensic_Detection', 'Jigsaw', 'Relative_Depth', 'Spatial_Relation']

    if task_name == 'all':
        return subtasks
    elif task_name in subtasks:
        return [task_name]
    else:
        raise ValueError(f"Invalid task name for BLINK: {task_name}. Please choose from {subtasks} or 'all'.")