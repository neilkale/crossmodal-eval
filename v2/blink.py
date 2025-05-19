import os
import jsonlines
from tqdm import tqdm
from task import Task

### TODO: Placeholder for BLINK class
class BLINK(Task):

    def __init__(self, *args, **kwargs):
        super().__init__(task_name='BLINK', *args, **kwargs)
        self.subtasks = []

    def get_subtasks(self, task_name):
        """
        Get the subtasks for a given task name.
        """
        if task_name == 'all':
            return self.subtasks
        elif task_name in self.subtasks:
            return [task_name]
        else:
            raise ValueError(f"Invalid task name for HardBLINK: {task_name}. Please choose from {self.subtasks} or 'all'.")