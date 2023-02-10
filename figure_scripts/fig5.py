#65;5802;1c
import svgutils.transform as sg
import sys
from subprocess import run

from svgutils.compose import Unit

from lxml import etree

from pathlib import Path

ovWdth = Unit('260mm')
ovHght = Unit('300mm')

output_path = '../output/figures/'
data_input_path = '../output/data_output/'
sim_input_path = '../output/simulation_figures/'

Path(output_path).mkdir(parents=True, exist_ok=True)

def generate_figure(input_data_prefix, input_sim_prefix, output_fn):
    fig0 = sg.SVGFigure(ovWdth,ovHght) 

    MPL_SCALE = 0.4

    HS = 16
    
    fig0.append(sg.FigureElement(etree.Element('rect', width='100%', height='100%', fill='white')))
    
    
    def get_file(fn, scale=MPL_SCALE, pos=None):
        print('get ', fn)
        fig = sg.fromfile(fn)
        plot = fig.getroot()
        plot.scale(scale, scale)
        if pos is not None:
            plot.moveto(*pos)
        return plot


    YS = 150
    YS = 150
    XS = 200

    dp = input_data_prefix
    sp = input_sim_prefix
    
    col1 = [
        [ [ dp + 'pch2_cell_co_number.svg', sp + 'cell_co_num.svg'], 'B' ],
        [ [ dp + 'pch2_segment_co_number.svg', sp + 'segment_co_num.svg'], 'C' ],
        [ [ dp + 'pch_hist.svg', sp + 'rel_pos.svg'], 'D' ],
    ]

    col2 = [
       [ [ dp + 'pch2_spacing.svg', sp + 'co_spacing.svg'], 'E' ],
        [ [ dp + 'pch_len_by_n.svg', sp + 'violin_L_n.svg'], 'F' ],
        [ [ dp + 'pch2_single_int_vs_L.svg', sp + 'single_int_vs_L_downsample.svg'], 'G' ],
 
    ]



    def make_row(data, label=True, ls=12):

        panels = []
        if label:
            for i,f in enumerate(data[0]):
                panels.append(get_file(f, pos=(i*XS, 0)))
            if data[1]:
                panels.append(sg.TextElement(-10, 10, data[1], size=20, weight="bold"))
        else:
            for i,f in enumerate(data):
                panels.append(get_file(f, pos=(i*XS, 0)))        
        return sg.GroupElement(panels)


    def make_group(data, label=True, ls=12):
        panels = []
        for i,r in enumerate(data):
            g = make_row(r, label, ls=ls)
            g.moveto(0, i*YS)
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


    # load matpotlib-generated figures

    g = make_group(col1)

    g.moveto(40, 50)

    head1_1 = sg.TextElement(25, 0, 'Experimental data', size=HS)
    head1_2 = sg.TextElement(XS+0+10, 0, 'Simulation output', size=HS)
    gh = sg.GroupElement([head1_1, head1_2])
    gh.moveto(40, 20)


    g2 = make_group(col2)

    g2.moveto(2*XS+40, 50)

    head2_1 = sg.TextElement(25, 0, 'Experimental data', size=HS)
    head2_2 = sg.TextElement(XS+0+10, 0, 'Simulation output', size=HS)
    gh2 = sg.GroupElement([head2_1, head2_2])
    gh2.moveto(2*XS+40, 20)


    g_sim = sg.GroupElement([g, gh, g2, gh2])
    g_sim.scale(0.3,0.3)
    g_sim.moveto(0, 140)


    img_label = sg.TextElement(5, 0, 'A', size=20, weight='bold')
    img_label.scale(0.3, 0.3)
    img_label.moveto(5,10)
    
    img = get_file('../extra_panels/pch2 image panel.svg', scale=1)
    img.moveto(0, 0)

    g_img = sg.GroupElement([img_label, img])
    
    g_page = sg.GroupElement([g_sim, g_img])
    
    
    fig0.append(g_page)

    fig0.save(output_fn)


generate_figure(data_input_path, sim_input_path+'pch_', output_path+'fig5.svg')
run(['inkscape', '-D', '-d', '300',  output_path+'fig5.svg', '-o', output_path+'fig5.png'])
run(['inkscape', '-D', '-d', '300',  output_path+'fig5.svg', '-o', output_path+'fig5.pdf'])
