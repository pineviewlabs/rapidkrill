#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
It contains modules for reporting RapidKrill results.

Created on Wed Jul 17 14:59:49 2019
@author: Alejandro Ariza, British Antarctic Survey
"""

# Import modules
import os, logging, logging.config, io, sendgrid, base64
from sendgrid.helpers.mail import Mail, Email, Content, Attachment
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from echopy.cmaps import cmaps
import toml

# Explicitly register pandas datetime converter for matplotlib
# It prevents Pandas FutureWarnings to pop up in the console
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# Log events while running
logger = logging.getLogger()
logging.config.fileConfig(os.path.join(os.path.dirname(__file__),
                                       '..','rapidkrill','logging.conf'))
def get_results(pro):
    # Load processed data variables
    transect    = pro['transect']
    t120r       = pro['t120r']
    t120intrvls = pro['t120intervals']
    nm120r      = pro['nm120r']
    lon120r     = pro['lon120r']
    lat120r     = pro['lat120r']
    sbline120r  = pro['sbliner'][0,:]
    NASC120swr  = pro['NASC120swr'][0,:]
    pc120swr    = pro['pc120swr'][0,:]

    # Build summary results
    return {'Time': np.array(t120r, dtype=str),
           'Longitude': np.round(lon120r, 5),
           'Latitude': np.round(lat120r, 5),
           'Transect': np.ones(len(t120r), dtype=int)*transect,
           'Miles': nm120r,
           'Seabed': np.round(sbline120r , 1),
           'NASC': np.round(NASC120swr, 2),
           '% samples': np.round(pc120swr,1)}

def log(pro, outputdir, savepng=True):
    """
    Log processed data (*.csv) and echograms (*.png) in rapidkrill/log/.

    Args:
        pro       (dict): processed data output from "process" routine.
        outputdir (str ): directory name under which log results will be saved.
        savepng   (bool): True to save echogram images, False to skip it.
    """

    results_data = get_results(pro)
    results = pd.DataFrame(results_data, columns= ['Time'     , 'Longitude',
                                                   'Latitude' , 'Transect' ,
                                                   'Miles'    , 'Seabed'   ,
                                                   'NASC'     , '% samples'])

    # Create new log subdirectory
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    csvfile = os.path.join(outputdir, 'output.csv')
    # Write results in CSV log file
    with open(csvfile, 'a') as f:
        results.to_csv(csvfile, index=False, mode='a',
                       header=f.tell()==0)

    # save png image
    if savepng:
        rawfiles    = pro['rawfiles']
        transect    = pro['transect']
        t120        = pro['t120']
        t120r       = pro['t120r']
        t120intrvls = pro['t120intervals']
        nm120r      = pro['nm120r']
        r120        = pro['r120']
        Sv120       = pro['Sv120']
        Sv120sw     = pro['Sv120sw']
        NASC120swr  = pro['NASC120swr'][0,:]

        # set figure
        plt.close()
        plt.figure(figsize=(8, 8))
        plt.subplots_adjust(left=0.066, right=1.055, bottom=0.065, top=0.985,
                            wspace=0, hspace=0.05)
        plt.rcParams.update({'font.size': 9, 'lines.linewidth': 1})

        # plot raw echogram
        plt.subplot(211).invert_yaxis()
        im=plt.pcolormesh(t120, r120, Sv120,
                          vmin=-80, vmax=-50, cmap=cmaps().ek500)
        plt.colorbar(im).set_label('Sv raw (dB re 1m$^{-1}$)')
        plt.gca().set_ylim(270,0)
        plt.gca().set_ylabel('Depth (m)')
        plt.gca().set_xlim(t120intrvls[0], t120intrvls[-1])
        plt.gca().set_xticks(t120intrvls[[0,-1]])
        plt.tick_params(labelright=False, labelbottom=False)

        # plot processed echogram
        ax= plt.subplot(212)
        ax = [ax, ax.twinx()]
        im=ax[0].pcolormesh(t120, r120, Sv120sw,
                            vmin=-80,vmax=-50, cmap=cmaps().ek500)
        plt.colorbar(im).set_label('Sv pro (dB re 1m$^{-1}$)')
        ax[0].invert_yaxis()
        ax[0].set_ylim(270,0)
        ax[0].set_ylabel('Depth (m)')

        # overlay distance/NASC info
        for t, nm, NASC in zip(t120r, nm120r, NASC120swr):
            ax[1].plot([t, t], [0, 1], color=[0,.8,0], linewidth=2)
            ax[1].text(t, .95, ' ' + str(transect) + ': ' + str(round(nm,2)),
                       fontweight='bold', color=[0,.8,0])
            ax[1].text(t, .02, ' ' + str(round(NASC,2)),
                       fontweight='bold', color=[1,0,0])
        ax[1].set_ylim(0, 1)
        ax[1].set_xlim(t120intrvls[0], t120intrvls[-1])
        ax[1].set_xticks(t120intrvls[[0,-1]])
        ax[1].tick_params(labelright=False)
        ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%d%b-%H:%M:%S'))

        # save figure
        pf = rawfiles[0].split('-')[0]
        fn = pd.to_datetime(str(t120[0])).strftime(pf + '-D%Y%m%d-T%H%M%S')
        imagefile = os.path.join(outputdir, fn+'.png')
        plt.savefig(imagefile,figsize=(8, 8), dpi=100)
        plt.close()

def console(pro):
    """
    Print summary report in the console while running RapidKrill. Data is
    retrieved from the "process" routine output, and is displayed as a table.

    Args:
        pro (dict): processed data output from "process" routine.
    """
    transect = pro['transect'  ]
    nm       = pro['nm120r'    ].flatten()#[pro['m120swr_'][0,:]]
    t        = pro['t120r'     ].flatten()#[pro['m120swr_'][0,:]]
    sb       = pro['sbliner'   ].flatten()#[0][:-1]
    NASC     = pro['NASC120swr'].flatten()#[pro['m120swr_']     ]
    pc       = pro['pc120swr'  ].flatten()#[pro['m120swr_']     ]

    # Preallocate table object
    table = io.StringIO()

    # Outline alignment and format for table lines, header, and data
    line   = '+{:-^10}+{:-^11}+{:-^25}+{:-^8}+{:-^13}+{:-^11}+ \n'
    header = '{:<9} | {:<9} | {:<23} | {:>6} | {:>11} |{:>12} \n'
    data   = '| {:<3d}      | {:<9.3f} | {:<15} | {:>6.1f} | {:>11.2f} | {:>9.1f} | \n'

    # Write table lines and header
    table.write(line.format('','','','','',''))
    table.write(header.format('| Transect','N. miles','Time','Seabed','NASC','% samples |'))
    table.write(line.format('','','','','',''))

    # Populate table with data
    for nmi, ti, sbi, NASCi, pci in zip(nm, t, sb, NASC, pc):
        table.write(data.format(transect, nmi, ti, sbi, NASCi, pci))

    # Close table with a line
    table.write(line[:-2].format('','','','','',''))

    # Print table in the console
    table = table.getvalue()
    print(table)
