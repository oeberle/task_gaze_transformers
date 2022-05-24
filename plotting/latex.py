
import pandas as pd
import matplotlib as mpl
import matplotlib.cm as cm

import os

class LatexHeatmap:
    """
    Args:
        vcenter:
            If not None, will be passed to `matplotlib.colors.TwoSlopeNorm`. 
            Otherwise `matplotlib.colors.Normalize` will be used to normalize 
            values. 
        min_max_factor (float, optional):
            vmin and vmax that will be passed to matplotlibs norm will be 
            multiplied by this value. To be used to make colors lighter
            (values >1).
        booktab_rules (bool, optional):
            If True, `\toprule` etc will be used (as output by 
            `pandas.DataFrame.to_latex`). If False, these rules will be 
            replaced by `\hline`.
            
    Details:
        Requires latex package colortbl. If `booktab_rules=True`, the package 
        booktabs is required as well.
    """
    def __init__(self, cmap=cm.PiYG, vcenter: float=0, min_max_factor=1, booktab_rules=True):
        self.cmap = cmap
        self.vcenter = vcenter
        self.min_max_factor = min_max_factor
        self.booktab_rules = booktab_rules
        self.float_format = '{:0.2f}'
    
    def from_dataframe(self, df: pd.DataFrame) -> str:
        df = df.copy()
        column_format = "l" + "".join(["r"]*df.shape[1])
        vmin = self.min_max_factor * df.min(numeric_only=True).min()
        vmax = self.min_max_factor * df.max(numeric_only=True).max()
        
        f = lambda x: self._make_latex_cell_code(
            x, float_format=self.float_format,  vmin=vmin, vmax=vmax, cmap=self.cmap, vcenter=self.vcenter)
        df = df.applymap(f)
        
        ltx = df.to_latex(column_format=column_format, index=False)
        ltx = self._recover_control_sequences(ltx)
        if not self.booktab_rules:
            ltx = self._replace_rules_by_hlines(ltx)
        return ltx
    
    @staticmethod
    def _make_latex_cell_code(x: float,float_format:str, **kwargs) -> str:
        if LatexHeatmap._is_numeric(x):
            rgba = LatexHeatmap._choose_color(x, **kwargs)
            rgb_code = LatexHeatmap._make_rgb_hex_code(*rgba[:3])
            s = r"\cellcolor[HTML]{" + rgb_code + "} " +  float_format.format(x)
            return s
        else:
            return x
    
    @staticmethod
    def _is_numeric(x) -> bool:
        try:
            x / 2
            int(x)
            return True
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _recover_control_sequences(s: str) -> str:
        s = s.replace(r"\}", "}")
        s = s.replace(r"\{", "{")
        s = s.replace(r"textbackslash ", "")
        return s

    @staticmethod
    def _replace_rules_by_hlines(latex_code: str) -> str:
        for loc in ["top","mid","bottom"]:
            latex_code = latex_code.replace(f"\\{loc}rule", r"\hline")
        return latex_code
    
    @staticmethod
    def _choose_color(x, vmin, vmax, vcenter=None, cmap=cm.hot) -> str:

        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax) \
               if vcenter is None \
               else mpl.colors.TwoSlopeNorm(vmin=vmin, vmax=vmax, 
                                            vcenter=vcenter)

        m = cm.ScalarMappable(norm=norm, cmap=cmap)
        return m.to_rgba(x)
    
    @staticmethod
    def _make_rgb_hex_code(r: float, g: float, b: float) -> str:
        r = f"{int(r*255):02X}"
        g = f"{int(g*255):02X}"
        b = f"{int(b*255):02X}"
        return r+g+b