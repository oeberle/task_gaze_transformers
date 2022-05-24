import numpy as np
import re

from typing import Tuple, Iterable, List, Union, Optional, Any


def assign_points_to_rect_areas(points: List[Tuple[float,float]], 
                                rect_areas: List[Tuple[float,float,float,float]]
                               ) -> List[Union[int, None]]:
    """
    Determines inside which rectangle(s) each point lies, if in any.

    Args:
        points (list): 
            List of 2-tuples of floats, where each tuple describes a point, 
            i.e. `(x,y)`. 
        rect_areas (list): 
            List of 4-tuples of floats, where each tuple where each tuple 
            describes a rectangle, i.e. `(x_left, x_right, y_top, y_bottom)`.

    Returns:
        List[Union[int, List[int], None]]: 
            For each element `p` in `points`:
                int: index of rectangle in `rect_areas` within which `p` lies, 
                     if `p` does not lie in any other rectangle.
                List[int]: indices of all rectangles, `p` lies in.
                None: if `p` does not lie in any rectangle.
    """
    rect_ids_for_each_point = [None]*len(points)
    for point_id, point_val in enumerate(points):
        x, y = point_val
        for rect_id, rect_val in enumerate(rect_areas):
            if isin_rect(x, y, rect_val):
                if rect_ids_for_each_point[point_id] is None:
                    rect_ids_for_each_point[point_id] = rect_id
                else: # make list of rect_ids
                    if type(rect_ids_for_each_point[point_id]) is not list:
                        rect_ids_for_each_point[point_id] = \
                            [rect_ids_for_each_point[point_id]]
                    rect_ids_for_each_point[point_id].append( rect_id )
    return rect_ids_for_each_point

def isin_rect(x: float, y: float, rect: Tuple[float,float,float,float]) -> bool:
    """
    Returns whether a point (x,y coordinate) is within an area definded by a 
    rectangle.

    Args:
        x (float): x position
        y (float): y position
        rect (tuple): 4-tuple of floats, `(x_left, y_top, x_right, y_bottom)`

    Returns:
        bool: True, if (x,y) lies inside the rectangle described by `rect`.
    
    Details:
        y-axis can be both inverted (0 at top) or normal (0 at bottom).
        x-axis has to "start" on the left hand side.
    """
    x_left, y_top, x_right, y_bottom = rect
    
    within_x_limits = (x <= x_right) & (x >= x_left)
    
    # check for y has to work for both inverted and normal y-axes 
    # (0,0 at top or at bottom)
    y_lower_value, y_greater_value = (np.min([y_top, y_bottom]), 
                                      np.max([y_top, y_bottom]))
    within_y_limits = (y <= y_greater_value) & (y >= y_lower_value)

    return within_x_limits & within_y_limits

def like(x: str, pattern: str) -> bool:
    """
    Returns True, if `x` matches the regex `pattern`, and False otherwise.
    """
    return re.search(pattern, x) is not None

def extract(x: str, pattern: str) -> str:
    """
    Extracts the sequence in `x` matched by regex ` pattern`, or None if no 
    match was found.
    """
    match_obj = re.search(pattern, x)
    if match_obj is None:
        return None
    else:
        return match_obj.group()
