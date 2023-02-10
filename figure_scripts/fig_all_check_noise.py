#65;5802;1c
import svgutils.transform as sg
import sys
from subprocess import run

from svgutils.compose import Unit

from lxml import etree

from pathlib import Path

ovWdth = Unit('250mm')
ovHght = Unit('290mm')

output_path = '../output/figures/'
sim_input_path = '../output/simulation_figures/'

Path(output_path).mkdir(parents=True, exist_ok=True)

def generate_figure(input_comp_prefixes, output_fn):
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

    OY = 0

    def make_col(data, label=True, ls=12):

        panels = []
        if label:
            for i,f in enumerate(data[0]):
                panels.append(get_file(f, pos=(0, i*YS)))
#                if data[1]:
#                    panels.append(sg.TextElement(-10, 10, data[1], size=20, weight="bold"))
                

                #            if data[1]:
#                l = sg.TextElement(0, 0, data[1], size=12, weight="bold", anchor="middle")
#                l.rotate(-90)
#                l.moveto(-20, YS/2-10)
#                panels.append(l)
#            if len(data)>2 and data[2]:
#                panels.append(sg.TextElement(XS, 7, data[2], size=12, anchor="middle"))
        else:
            for i,f in enumerate(data):
                panels.append(get_file(f, pos=(0, i*YS)))        
        return sg.GroupElement(panels)


    def make_group(data, label=True, ls=12):
        panels = []
        for i,r in enumerate(data):
            g = make_col(r, label, ls=ls)
            g.moveto(i*XS, 0)
            panels.append(g)

        return sg.GroupElement(panels)
    #         

    def make_single(data, label=True, label_offset_x=-10):
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


    all_g = []


    plot_grid = [ [ cp + '_cell_co_num.svg', cp + '_segment_co_num.svg', cp+ '_co_spacing.svg', cp + '_rel_pos.svg', cp + f'_rel_pos_{hists[0]}.svg', cp + f'_rel_pos_{hists[1]}.svg' ]  for cp, cl, hists in input_comp_prefixes ] 

    g = make_group(plot_grid, label=False)
    
    g.moveto(40, 40)

    top_labels = sg.GroupElement([ sg.TextElement(idx*XS + 0.45*XS, 10, l, size=20, anchor='middle') for (idx, l) in enumerate(['WT', 'WT+noise', 'OX', 'OX+noise' ]) ])

    top_labels.moveto(40,20)

    row_labels = sg.GroupElement([ sg.TextElement(0, 10+YS*idx, l, size=20, anchor='middle', weight='bold') for (idx, l) in enumerate(['A', 'B', 'C', 'D', 'E', 'F' ]) ])
    row_labels.moveto(15,30)
    
    all_g = [g, top_labels, row_labels]
    
    gpage = sg.GroupElement(all_g)
    gpage.scale(0.3,0.3)
    fig0.append(gpage)

    fig0.save(output_fn)


prefixes = [('test_5_1_0.0', 'WT', (1,2)), ('test_5_1_0.2', 'WT+noise', (1,2)), ('test_5_4.5_0.0', 'OX', (2,3)), ('test_5_4.5_0.2', 'OX+noise', (2,3))]
generate_figure([(sim_input_path + p[0], p[1], p[2]) for p in prefixes], output_path+'fig_all_noise.svg')
run(['inkscape', '-D', '-d', '300',  output_path+'fig_all_noise.svg', '-o', output_path+'fig_all_noise.png'])

