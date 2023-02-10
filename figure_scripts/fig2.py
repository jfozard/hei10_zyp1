#65;6003;1c#65;5802;1c
import svgutils.transform as sg
import sys
from subprocess import run

from svgutils.compose import Unit
from lxml import etree

ovWdth = Unit('110mm')
ovHght = Unit('170mm')



def generate_figure(exp_data_path, input_prefix, output_fn):
    fig0 = sg.SVGFigure(ovWdth,ovHght) 

    MPL_SCALE = 0.4
    JL_SCALE = 0.02

    fig0.append(sg.FigureElement(etree.Element('rect', width='100%', height='100%', fill='white')))
    
    def get_file(fn, scale=MPL_SCALE, pos=None):
        print('get', fn)
        fig = sg.fromfile(fn)
        plot = fig.getroot()
        plot.scale(scale, scale)
        if pos is not None:
            plot.moveto(*pos)
        return plot


    YS = 160
    XS = 200

    ch_labels = [ "Shortest" , "", "", "", "Longest"]


    row1 = [ [ ( exp_data_path+'mlh_data.svg', input_prefix+'-cell-number.svg', exp_data_path+'expt_univalents.svg', input_prefix+'-cell-no-univalents.svg', input_prefix+'_kymo.svg'
    ) ] ,
             [ [ (input_prefix+f'-ch-{i}-number.svg') for i in range(5)] ] ] 

    

    def make_col(data, label=True, ls=12):

        panels = []
        if label:
            for i,f in enumerate(data[0]):
                panels.append(get_file(f, pos=(0, i*YS)))
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


    # load matpotlib-generated figures

    g = make_group(row1)

    g.moveto(40,30)

    ox = 30
    oy = 0
    
    head = [ sg.TextElement(ox, oy, 'Experimental data', size=14),
             sg.TextElement(ox, oy+YS, 'Simulation output', size=14),
             sg.TextElement(ox, oy+2*YS, 'Experimental data', size=14),
             sg.TextElement(ox, oy+3*YS, 'Simulation output', size=14),
             sg.TextElement(ox, oy+4*YS, 'Simulation output', size=14) ]

    
    gh = sg.GroupElement(head)
    gh.moveto(40, 30)

    head2 = [ sg.TextElement(ox+XS, oy, 'Simulation output', size=14) ]
    

    gh2 = sg.GroupElement(head2)
    gh2.moveto(40, 30)

    
    gpage = sg.GroupElement([g, gh, gh2])
    gpage.scale(0.2,0.2)

    goverlay = get_file('../extra_panels/vert_arrow_overlay.svg', scale=1)

    fig0.append(gpage)
    fig0.append(goverlay)
 

    # save generated SVG files
    fig0.save(output_fn)



output_path = '../output/figures/'    
exp_data_path = '../output/data_output/'
simulation_figures_path = '../output/simulation_figures/'

prefix = 'regtot-nuc-poisson-0.0'
generate_figure(exp_data_path, simulation_figures_path+prefix, output_path+'fig2.svg')
run(['inkscape', '-D', '-d', '300',  output_path+'fig2.svg', '-o', output_path+'fig2.png'])
run(['inkscape', '-D', '-d', '300',  output_path+'fig2.svg', '-o', output_path+'fig2.pdf'])
