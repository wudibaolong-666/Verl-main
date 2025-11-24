import json


class Prompt:
    def __init__(self):
        self.sp = """
    You are a grid-based cave exploration decision-making agent. 
    The following provides an overview of the environment, along with the details of the input information and the required output format you must strictly follow.  
    
     1. Introduction 
        This is a grid-based cave of size x by size, with one gold item and multiple pits. 
        The gold and pits are placed in separate, non-overlapping cells. 
        The agent must explore the cave and accomplish the following objectives:
        (1) Primary Objective: First, collect the gold. Then, use the visited positions to safely navigate back to the starting point.
        (2) Secondary Optimization:  Minimize the number of moves taken.
        (3) Failure Conditions: The agent fails if it falls into a pit.

     2. Action Space:
        The agent can move in four directions: <MoveUp>, <MoveDown>, <MoveLeft>, and <MoveRight>. 
        These actions will change the agent's position on the grid according to the following movement rules:

     3. Coordinate System Explanation
        Coordinate Range: The coordinates (x, y) range from (1, 1) to (size, size), where size represents the size of the grid.
        Movement Rules:
            (1) <MoveUp>: Increases the y-coordinate by 1 (e.g.,(1, 1) → (1, 2)), but cannot be valid when y = size. 
            (2) <MoveDown>: Decreases the y-coordinate by 1 (e.g.,(2, 2) → `(2, 1)), but cannot be valid when y = 1.
            (3) <MoveLeft>: Decreases the x-coordinate by 1 (e.g.,(3, 1) → (2, 1)), but cannot be valid when x = 1.
            (4) <MoveRight>: Increases the x-coordinate by 1 (e.g.,(1, 2) → (2, 2)), but cannot be valid when x = size.
        Adjacency Definition: 
            A position is adjacent to another if it can be reached by one of the four actions without going out of bounds.
            For size = 3:
                From (2, 2), the adjacent positions are (2, 3), (2, 1), (1, 2), and (3, 2).
                From (1, 1), the adjacent positions are (1, 2) and (2, 1).

     4. Environment Elements:
        pit: A dangerous area that causes the agent die and terminate the game.
        gold: The agent must find and collect the gold. Once the agent reaches the gold’s location, the gold is collected.

     5. Input:
        "grid-based cave size": The size of the cave (e.g., 3 means it’s a 3x3 grid with 9 cells).
        "num of pit":  The number of pits in the cave (e.g., 1 means there is 1 pit to avoid).
        "Start position": The agent's starting point, which must be returned to after collecting the gold.
        "Current position": The agent’s current position in the grid (e.g., (x, y):(1, 1)).
        "previous action":  Refers to the action the agent take in the previous step.
        "gold_collected": "No" means the agent needs to find the gold first. "Yes" means the gold has been found, and the agent must return to the start
        "breeze": "No" means there is no pit in the adjacent cells to the agent's current position. "Yes" means one of the adjacent cells contains a pit.
        "glitter": "No" means there is no gold in the adjacent cells to the agent's current position. "Yes" means one of the adjacent cells contains a gold.
        "bump":  "No" means the previous action is a valid move. "Yes" means the previous action is an invalid move and should choose a valid action.
        "visited breeze positions": positions where breeze was detected.
        "visited glitter positions": positions where glitter was detected.
        "visited positions": All positions the agent has visited

     6. Output:
        First, update the previous positions based on the input and avoid repeating positions.
        Then, perform step-by-step analysis:
          (1) ......
          (2) ......
        After analysis, output the updated previous positions and Action(strictly follow the format below):
          Previous positions updated:{
            "visited breeze positions": [(...), (...)],
            "visited glitter positions": [(...), (...)],
            "visited positions": [(...), (...)]
          },
          Action: <ActionName>
     """

        input_json = {
            'grid-based cave size': 3,
            'num of pit': 1,
            "Start position(x, y)": (1, 1),
            "Current position(x, y)": (2, 1),
            "previous action": "<MoveRight>",
            "gold_collected": "No",
            "breeze": "No",
            "glitter": "Yes",
            "bump": "No",
            "Previous positions": {
                "visited breeze positions": [],
                "visited glitter positions": [],
                "visited positions": [(1, 1)]
            },
        }
        self.eip = f"""
        The input information are as follows:{input_json}
    """

        previous_json = {
            "visited breeze positions": [],
            "visited glitter positions": [(2, 1)],
            "visited positions": [(1, 1), (2, 1)]
        }
        guess_json = {
            "pit": [],
            "gold": [(2, 2), (3, 1)],
        }
        self.eop = f"""
    Update the Previous positions:
      The current position (2, 1) is not in the visited positions,  so it should be added to visited positions.
      The breeze is 'No', there is no need to add the current position (2, 1) to visited breeze positions.
      The glitter is 'Yes', and the current position (2, 1) is not in visited glitter positions, so it should be added to visited glitter positions.
      so the updated positions:{previous_json}
    Analysis: 
      If gold_collected is 'No', it means the agent needs to find the gold first.
      At the current position (2, 1), with the breeze being 'No', the adjacent in-bounds positions are (1, 1), (3, 1), and (2, 2), all of which are safe.
      Since the glitter is 'Yes', the potential gold positions in-bounds are (1, 1), (3, 1), and (2, 2).
      If (1, 1) is the gold position, and the agent has already visited (1, 1), the gold should have been collected. 
      Therefore, the remaining potential gold positions are (3, 1) and (2, 2).
      We randomly choose position (3, 1) as the target, and the required action is <MoveRight>.
    Previous positions updated:{previous_json}
    Action:<MoveRight>
    """

    def wrap_message(self, obs):
        obs_prompt = (obs +
                      "\nFor the output, you should first update the Previous Positions, then perform the analysis step by step. "
                      + "Finally, output the updated Previous Positions and Action:\n"

                        "Previous positions updated: {\n"
                        '  "visited breeze positions": [(...), (...)],\n'
                        '  "visited glitter positions": [(...), (...)],\n'
                        '  "visited positions": [(...), (...)]\n'
                        "},\n"

                        'Action: <ActionName>.\n'
                      )

        message = [
            {"role": "system", "content": self.sp},  # 系统的提示，通常用于设置模型的行为规范或提供背景信息
            # {"role": "user", "content": self.eip},  # 用户的输入，表示模型要响应的实际问题或观察
            # {"role": "assistant", "content": self.eop},  # 模型（助手）的响应示例
            {"role": "user", "content": obs_prompt}
        ]
        return message


if __name__ == "__main__":
    prompt = Prompt()
    print(prompt.sp)  # 系统提示
    print(prompt.eip)  # example_input
    print(prompt.eop)  # example_output