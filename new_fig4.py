


import svgutils.transform as sg
import sys
from subprocess import run

from svgutils.compose import Unit

from lxml import etree

ovWdth = Unit('210mm')
ovHght = Unit('150mm')



def generate_figure(input_prefix, output_fn):
    fig0 = sg.SVGFigure(ovWdth,ovHght) 

    MPL_SCALE = 0.4
    JL_SCALE = 0.02

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

    ch_labels = [ "Shortest" , "", "", "", "Longest"]

    row1 = [ [ [ 'wt_pos_all.svg', 'pos_all.svg', prefix + '-relpos.svg' ] ],
             [ [ 'wt_spacing.svg', 'spacing.svg', prefix + '-spacing.svg' ] ],
             [ [ 'wt_intensity_violin.svg', 'intensity_violin.svg',  prefix+ '-violin_intensities.svg' ] ],
           ]    

    def make_row(data, label=True, ls=12):

        panels = []
        if label:
            for i,f in enumerate(data[0]):
                panels.append(get_file(f, pos=(i*XS, 0)))
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

    g = make_group(row1)

    g.moveto(40, 50)

    
    head1 = sg.TextElement(XS/2, 0, 'Wild-type data', size=16, anchor="middle")

    head2 = sg.TextElement(3*XS/2-10, 0, '', size=16, anchor="middle")
    e = etree.Element('tspan')
    e.set('font-style','italic')
    e.text = "zyp1a-2/zyp1b-1  "
    head2.root.append(e)
    f = etree.Element('tspan')
    f.text = " data"
    
    head2.root.append(f)
    head3 = sg.TextElement(5*XS/2-10, 0, 'Simulation output', size=16, anchor="middle")
    gh = sg.GroupElement([head1, head2, head3])
    gh.moveto(40, 30)


    glabels = sg.GroupElement([ sg.TextElement(u[0], u[1], u[2], size=16) for u in [(0,-20,'A'), (0, YS, 'B'), (0, 2*YS, 'C') ] ])
    glabels.moveto(20, 60)
    
    gpage = sg.GroupElement([g, gh, glabels])
    gpage.scale(0.3,0.3)
    fig0.append(gpage)

    # save generated SVG files
    fig0.save(output_fn)


for prefix in ['poisson-0.0']:
    generate_figure(prefix, 'fig4-'+prefix+'.svg')
    run(['inkscape', '-D', '-d', '300',  'fig4-'+prefix+'.svg', '-o', 'fig4-'+prefix+'.png'])
