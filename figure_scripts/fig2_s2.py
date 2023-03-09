#65;5802;1c
import svgutils.transform as sg
import sys
from subprocess import run

from svgutils.compose import Unit

from lxml import etree

from pathlib import Path

ovWdth = Unit('230mm')
ovHght = Unit('70mm')

output_path = '../output/figures/'
data_input_path = '../output/data_output/'
sim_input_path = '../output/simulation_figures/'

Path(output_path).mkdir(parents=True, exist_ok=True)

def generate_figure(input_sim_prefix, output_fn):
    fig0 = sg.SVGFigure(ovWdth,ovHght) 

    MPL_SCALE = 0.4

    HS = 16

    fig0.append(sg.FigureElement(etree.Element('rect', width='100%', height='100%', fill='white')))
    
    def get_file(fn, scale=MPL_SCALE, pos=None):
        fig = sg.fromfile(fn)
        plot = fig.getroot()
        plot.scale(scale, scale)
        if pos is not None:
            plot.moveto(*pos)
        return plot


    YS = 150
    YS = 150
    XS = 200

    sp = input_sim_prefix
    

    def make_single(data, label=True, label_offset_x=-30):
        panels = []
        if label:
            f, l  = data
            panels.append(get_file(f, pos=(0, 0)))
            if l:
                panels.append(sg.TextElement(label_offset_x, 10, l, size=20, weight='bold'))
        else:
            f=data
            panels.append(get_file(f, pos=(0, 0)))        
        return sg.GroupElement(panels)



    
    # load matpotlib-generated figures

    gA = make_single((data_input_path + 'mlh_data_ox.svg','A'))

    gA.moveto(35, 10)

    gB = make_single((sim_input_path + 'regtot-nuc-ox-poisson-0.0-cell-number.svg','B'))

    gB.moveto(245, 10)

    gC = make_single((sim_input_path + 'regtot-nuc-ox-poisson-0.2-cell-number.svg', 'C'))

    gC.moveto(455, 10)

    gpage = sg.GroupElement([gA, gB, gC])
    gpage.scale(0.35,0.35)
    
    fig0.append(gpage)

    fig0.save(output_fn)


generate_figure(sim_input_path, output_path+'fig2_s2.svg')
run(['inkscape', '-D', '-d', '300',  output_path+'fig2_s2.svg', '-o', output_path+'fig2_s2.png'])
run(['inkscape', '-D', '-d', '300',  output_path+'fig2_s2.svg', '-o', output_path+'fig2_s2.pdf'])
