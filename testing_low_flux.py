# Imports
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.ticker as mtick
import matplotlib.patches as patches
import matplotlib
import shapely.geometry as geom
from shapely import affinity
from collections import Counter
import pickle
import time
import numpy as np
import subprocess
from copy import deepcopy
import os
import warnings
from collections import Counter
from matplotlib.patches import Rectangle, Ellipse
import struct
import matplotlib.pyplot as plt
import pandas as pd
import astropy.visualization as vis
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import skycoord_to_pixel, pixel_to_skycoord
from astropy.table import Table
from astropy import units as u
from subprocess import call
import sys
sys.path.append('/home/rafael/data/mostertrij/LOFAR-PINK-library2')
import pinklib.postprocessing
import seaborn as sns
from astropy.wcs import WCS
from shapely.geometry import Polygon
from shapely.ops import cascaded_union
import argparse
from scipy.ndimage import gaussian_filter
from astropy.visualization import (MinMaxInterval, SqrtStretch,
                                   ImageNormalize)

matplotlib.rcParams.update({'font.size': 18})

parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('-d','--debug', help='Enabling debug will render debug output and plots.',
        dest='debug', action='store_true', default=False)
parser.add_argument('-o','--overwrite', help='Enabling overwrite will overwrite pybdsf catalog for augmented cutouts.',
        default=False, action='store_true')
parser.add_argument('-n','--noises',nargs='+',type=int, help='List of noise sigmas used to augment cutout to resemble fainter sources.',
        default=[0,3,10], dest='noises')
#parser.add_argument('-g','--gaussian-blurs',nargs='+',type=int, help='List of Gaussian kernelsize used to augment cutout to resemble fainter sources.',
#        default=[0,5,15], dest='gaussian_blurs')
args = vars(parser.parse_args())

debug = args['debug']
overwrite = args['overwrite']
#gaussian_blurs = args['gaussian_blurs']
noises = args['noises']

if debug:
    print("Parsed arguments: Debug:", debug, "overwrite",overwrite, 'noises', noises)

if not debug:
    pd.set_option('mode.chained_assignment', None)
# ### Import data, set up file locations and names
# base path
cat_dir = '/home/rafael/data/mostertrij/data/catalogues'
# data directory
data_directory = '/home/rafael/data/mostertrij/data/LoTSS_DR2/RA0h_field/'
field = 'P11Hetdex12'
# comp_path=os.path.join(cat_dir,'LOFAR_HBA_T1_DR1_merge_ID_v1.2.comp.h5')
# gaul_path=os.path.join(cat_dir,'LOFAR_HBA_T1_DR1_catalog_v0.99.gaus.h5')
vac_path=os.path.join(cat_dir,'LOFAR_HBA_T1_DR1_merge_ID_optical_f_v1.2.h5')
vac = pd.read_hdf(vac_path,'df')
# Name of the fits file
fits_filename = 'mosaic-blanked.fits'
fits_rms_filename = 'mosaic.rms.fits'


def get_idx_dict(cat):
    """Create dict that returns row index of objects when given sourcename"""
    idx_dict = {s:idx for s, idx in zip(cat.Source_Name.values,cat.index.values)}
    return idx_dict

def get_comps(predicted_comp_cat):
    """Turn list of source_names, component_names into a list"""
    combined_names = list(OrderedDict.fromkeys(predicted_comp_cat['Source_Name'].values))
    comp_dict = {s:[] for s in combined_names}
    for s, comp in zip(predicted_comp_cat.Source_Name.values,predicted_comp_cat.Component_Name.values):
            comp_dict[s].append(comp)
    return [comp_dict[n] for n in combined_names]

def ellipse(x0,y0,a,b,pa,n=200):
    theta=np.linspace(0,2*np.pi,n,endpoint=False)
    st = np.sin(theta)
    ct = np.cos(theta)
    pa = np.deg2rad(pa+90)
    sa = np.sin(pa)
    ca = np.cos(pa)
    p = np.empty((n, 2))
    p[:, 0] = x0 + a * ca * ct - b * sa * st
    p[:, 1] = y0 + a * sa * ct + b * ca * st
    return Polygon(p)

class Make_Shape(object):
    '''Basic idea taken from remove_lgz_sources.py -- maybe should be merged with this one day
    but the FITS keywords are different.
    '''
    def __init__(self,clist):
        '''
        clist: a list of components that form part of the source, with RA, DEC, DC_Maj...
        '''
        ra=np.mean(clist['RA'])
        dec=np.mean(clist['DEC'])

        ellist=[]

        for ir, r in clist.iterrows():
            n_ra=r['RA']
            n_dec=r['DEC']
            x=3600*np.cos(dec*np.pi/180.0)*(ra-n_ra)
            y=3600*(n_dec-dec)
            newp=ellipse(x,y,r['DC_Maj']+0.1,r['DC_Min']+0.1,r['PA'])
            ellist.append(newp)
        self.cp=cascaded_union(ellist)
        self.ra=ra
        self.dec=dec
        self.h=self.cp.convex_hull
        a=np.asarray(self.h.exterior.coords)
        #for i,e in enumerate(ellist):
        #    if i==0:
        #        a=np.asarray(e.exterior.coords)
        #    else:
        #        a=np.append(a,e.exterior.coords,axis=0)
        mdist2=0
        bestcoords=None
        for r in a:
            dist2=(a[:,0]-r[0])**2.0+(a[:,1]-r[1])**2.0
            idist=np.argmax(dist2)
            mdist=dist2[idist]
            if mdist>mdist2:
                mdist2=mdist
                bestcoords=(r,a[idist])
        self.mdist2=mdist2
        self.bestcoords=bestcoords
        self.a=a

    def length(self):
        return np.sqrt(self.mdist2)

    def pa(self):
        p1,p2=self.bestcoords
        dp=p2-p1
        angle=(180*np.arctan2(dp[1],dp[0])/np.pi)-90
        if angle<-180:
            angle+=360
        if angle<0:
            angle+=180
        return angle

    def width(self):
        p1,p2=self.bestcoords
        d = np.cross(p2-p1, self.a-p1)/self.length()
        return 2*np.max(d)
        
def sourcename(ra,dec):
    sc=SkyCoord(ra*u.deg,dec*u.deg,frame='icrs')
    s=sc.to_string(style='hmsdms',sep='',precision=2)
    return str('ILTJ'+s).replace(' ','')[:-1]

def save_cat(dataframe, output_dir, save_name, keys=None):
    # Save to hdf
    hdf_path = os.path.join(output_dir, save_name + '.h5')
    dataframe.to_hdf(hdf_path,'df')
    # Save to fits
    fits_path = os.path.join(output_dir, save_name + '.fits')
    if os.path.exists(fits_path):
        print("Fits file exists. Overwriting it now")
        # Remove old fits file
        os.remove(fits_path)
    t = Table.from_pandas(dataframe)
    t.write(fits_path, format='fits')
    
# Testing the linking of flux in a noisified image to the unaltered image

# Load single field image
## Load LoTSS Lockman hole catalogue
LoTSS_field_path = os.path.join(data_directory, field, fits_filename)
image, hdr = pinklib.postprocessing.load_fits(LoTSS_field_path, dimensions_normal=True)
wcs = WCS(hdr,naxis=2)
restfreq = hdr['RESTFRQ']
angular_resolution = abs(hdr['CDELT1'])*3600
beam_size_in_arcsec = 6
# Calculate size of Gaussian kernel
gaussian_blur = pinklib.postprocessing.FWHM_to_sigma_for_gaussian(beam_size_in_arcsec*angular_resolution)


# Load single field cat

# Make large image cutout
## Choose patch of sky to focus on
loc = SkyCoord('11:35:12.17 +48:26:41.3', frame='icrs', unit=(u.hourangle, u.deg))
width_in_arcsec = 8*60
## Create cutout
subimage = pinklib.postprocessing.make_numpy_cutout_from_fits(
    width_in_arcsec, width_in_arcsec, loc.ra, loc.dec,
 LoTSS_field_path, dimensions_normal=True, just_data=False,
    arcsec_per_pixel=angular_resolution)
bb = subimage.bbox_original
if debug:
    print(f'Restfrequency in Hz:', int(restfreq))
    print(f'Angular resolution: {angular_resolution:.2f} arcsec/pixels')
    print(f'Cutout shape: {np.shape(subimage.data)} pixels')
w,h = np.shape(subimage.data)
## Get ra, dec extent of cutout
# ras, decs = wcs.all_pix2world([[0,0],[0,w],[0,int(w/2)],[int(w/2),0],[w,0],[w,w]],0).T
# minra, maxra = np.min(ras), np.max(ras)
# mindec, maxdec = np.min(decs), np.max(decs)
minra = np.min([loc.ra.value - width_in_arcsec/(3600*2),
                       loc.ra.value + width_in_arcsec/(3600*2)])
maxra = np.max([loc.ra.value - width_in_arcsec/(3600*2),
                       loc.ra.value + width_in_arcsec/(3600*2)])
mindec = np.min([loc.dec.value - width_in_arcsec/(3600*2),
                       loc.dec.value + width_in_arcsec/(3600*2)])
maxdec = np.max([loc.dec.value - width_in_arcsec/(3600*2),
                       loc.dec.value + width_in_arcsec/(3600*2)])

# Query sources within cutout from vac cat
field_cat = vac[vac.Mosaic_ID == field]
cutout_cat = field_cat[(field_cat.RA > minra) & (field_cat.RA < maxra) & 
                      (field_cat.DEC > mindec) & (field_cat.DEC < maxdec)]
# merge LGZ and pybdsf sizes
cutout_cat['source_length'] = cutout_cat.Maj.fillna(0) + cutout_cat.LGZ_Size.fillna(0)
cutout_cat['source_width'] = cutout_cat.Min.fillna(0) + cutout_cat.LGZ_Width.fillna(0)
cutout_cat['source_PA'] = cutout_cat.PA.fillna(0) + cutout_cat.LGZ_PA.fillna(0)


c = SkyCoord(cutout_cat.RA, cutout_cat.DEC, unit='deg')
pixels = skycoord_to_pixel(c,subimage.wcs, origin=0)

## Plot cutout
if debug:
    fig,ax = pinklib.postprocessing.plot_cutout2D(subimage.data, wcs=subimage.wcs, sqrt=True,
                                         colorbar=True,cmap='magma', return_fig=True);
    # Plot sources
    for i, (x,y) in enumerate(zip(*pixels)):
        ax.plot(x,y,'w', linestyle=None,marker='x')
        ax.text(x+5,y,i,c='k',fontsize=15) #shadow
        ax.text(x+4,y,i,c='w',fontsize=15)
        ax.set_title("Original cutout")
    plt.show()

original_flux = cutout_cat.Total_flux.sum()
stats = pd.DataFrame({'Cat':['Original'],
                      'Detected sources':[len(cutout_cat)],
                      'Linked sources':[len(cutout_cat)],
                     'Unassociated sources':[0],
                      'Detected flux (mJy)':[original_flux],
                      'Linked flux (mJy)':[np.nan],
                    'Unassociated flux (mJy)':[np.nan],
                      'Linked flux fraction':[np.nan],
                     'Unassociated flux fraction':[np.nan]})

# For all augmentations repeat this loop

for noise in noises:
    augmented_cat=pd.DataFrame()
    unassoc_cat=pd.DataFrame()
    retrieved_component_cat=pd.DataFrame()
    retrieved_source_cat=pd.DataFrame()
    plt.close("all")
    ## Augment cutout
    img = subimage.data
    normal = np.random.normal(0, np.std(img), np.shape(img))

    if noise > 0:
        gaussian_blur = pinklib.postprocessing.FWHM_to_sigma_for_gaussian(beam_size_in_arcsec*angular_resolution)
        #augmented_img = img + noise*normal
        #augmented_img = gaussian_filter(augmented_img, sigma=gaussian_blur)
        augmented_img = img + gaussian_filter(noise*normal, sigma=gaussian_blur)
    else:
        gaussian_blur=0
        augmented_img = img
    augmented_cutout_filename = f'augmented_cutout_blurred{gaussian_blur:.2f}sigma_noise{noise}sigma.fits'
    # Write the cutout to a new FITS file
    hdr.update(subimage.wcs.to_header())
    fits.writeto(augmented_cutout_filename, augmented_img, hdr,overwrite=True)



    ## Plot cutout
    if debug:
        fig,ax = pinklib.postprocessing.plot_cutout2D(augmented_img, wcs=subimage.wcs, sqrt=True,
                                             colorbar=True,cmap='magma', return_fig=True);
        # Plot sources
        for i, (x,y) in enumerate(zip(*pixels)):
            ax.plot(x,y,'w', linestyle=None,marker='x')
            ax.text(x+5,y,i,c='k',fontsize=15) #shadow
            ax.text(x+4,y,i,c='w',fontsize=15)
            ax.set_title(f"Cutout blurred {gaussian_blur:.2f}sigma, noise {noise}sigma")
        plt.show()

    # Run PyBDSF on cutout
    augmented_cat_fits_path = os.path.join(data_directory,field,f'augmented_cat_blurred{gaussian_blur:.2f}sigma_noise{noise}_sigma.fits')
    augmented_cat_flag_path = os.path.join(data_directory,field,f'augmented_cat_blurred{gaussian_blur:.2f}sigma_noise{noise}_sigma.flag')
    single_text = f"""
#Imports
import bdsf
img = bdsf.process_image('{augmented_cutout_filename}', thresh_isl=4.0, thresh_pix=5.0, 
                         rms_box=(160,50), rms_map=True, 
                         mean_map='zero',  ini_method='intensity', adaptive_rms_box=True, 
                         adaptive_thresh=150, rms_box_bright=(60,15), group_by_isl=False, 
                         group_tol=10.0, atrous_do=True,atrous_jmax=4, flagging_opts=True, 
                         flag_maxsize_fwhm=0.5,advanced_opts=True, blank_limit=None,
                         frequency={int(restfreq)})
# Get PyBDSF cat for cutout
## Write the source list catalog.
img.write_catalog(outfile='{augmented_cat_fits_path}',
              format='fits', catalog_type='srl', clobber={overwrite})
    """
                                            
    if overwrite or (not os.path.exists(augmented_cat_fits_path) and not os.path.exists(augmented_cat_flag_path)):
        with open('exec_singularity.py','w') as f:
            f.write(single_text)
        
        #exec singularity
        call("singularity exec /home/rafael/data/mostertrij/singularity/lofar_sksp_fedora27_ddf_msoverview.sif python exec_singularity.py", shell=True)


    #Linking catalogue A to noisy catalogue A* will proceed as follows:
    #i) For each source in A*, check if its RA,DEC falls within the ellips described by 
    #the RA,DEC, Positional Angle (PA) and length of the Major&Minor axis of a source in A.
    # If a source appears in multiple ellipses, it is assigned to the ellipse with the 
    # smallest number of matches.
    #ii) If no match is founds, register the flux of the source in A* as noise.
    #iii) If a single match is found, link the flux of the source in A* to that in A.
    #iv) If multiple matches are found, link the flux to the source in A* to the matched one in A that is closest in terms of RA,DEC.

    # Load cat for augmented image
    if not os.path.exists(augmented_cat_fits_path):
        print(f"PyBDSF source catalogue was not created for image blurred with {gaussian_blur:.2f}sigma and noise {noise}sigma.")
        print("Because PyBDSF found zero sources, or due to an error in sourcefinding.")
        augmented_cat = pd.DataFrame({'Total_flux':[0]})
        with open(augmented_cat_flag_path,'w') as f:
            f.write("")

    else:
        augmented_cat = pinklib.postprocessing.fits_catalogue_to_pandas(augmented_cat_fits_path)
        ca = SkyCoord(augmented_cat.RA, augmented_cat.DEC, unit='deg')
        ca_pixels = skycoord_to_pixel(ca,subimage.wcs, origin=0)

        if debug:
            ## Plot cutout
            fig,ax = pinklib.postprocessing.plot_cutout2D(augmented_img, wcs=subimage.wcs, sqrt=True,
                                                 colorbar=True,cmap='magma', return_fig=True);
        # Plot sources in vac and newly found sources
        es = []
        for i, (x,y, l,w,pa) in enumerate(zip(*pixels,
                                                     cutout_cat.source_length, 
                                        cutout_cat.source_width,
                                       cutout_cat.source_PA)):
            if debug:
                ax.plot(x,y,'k', linestyle=None,marker='x',markersize=15)
                ax.text(x-15,y,len(pixels[0])-i,c='k',fontsize=15) #shadow

            e = Ellipse((x,y),l,w,angle=pa-90, linewidth=2, edgecolor='k',facecolor='none')
            es.append(e)
        matches = [[] for e in es]
        for i, e in enumerate(es):
            #print(e.contains_points(list(zip(*ca_pixels))))
            for j, (x,y) in enumerate(zip(*ca_pixels)):
                if e.contains_point([x,y]):
                    matches[i].append(j)
            if debug:
                ax.add_patch(e)   
        matches = np.array(matches)

        ##### Remove duplicate matches
        len_matches = [len(l) for l in matches]
        len_matches_ranked = np.argsort(len_matches)
        c = Counter(list(pinklib.postprocessing.flatten(matches)))
        new_matches = [[] for e in es]

        dups = []
        for i, m in enumerate(matches[len_matches_ranked]):
            for mm in m:
                if c[mm] < 2:
                    new_matches[i].append(mm)
                    dups.append(mm)
                else:
                    if not mm in dups:
                        new_matches[i].append(mm)
                        dups.append(mm)
        new_matches = np.array(new_matches)
        unique_matches = list(set(pinklib.postprocessing.flatten(new_matches)))
        #print("heyho: new_matches", new_matches, "unique_matches", unique_matches)



        ################ Visualization
        if debug:
            for i, (x,y) in enumerate(zip(*ca_pixels)):
                ax.plot(x,y,'w', linestyle=None,marker='+')
                ax.text(x+5,y,i,c='k',fontsize=25) #shadow
                ax.text(x+4,y,i,c='w',fontsize=25)
            # Show matches as similarly colored
            prop_cycle = plt.rcParams['axes.prop_cycle']
            colors = prop_cycle.by_key()['color']
            for i, m in enumerate(new_matches):
                for mm in m:
                    ax.plot(ca_pixels[0][mm],ca_pixels[1][mm],c=colors[i%10], linestyle='None',
                            marker='+',markersize=15)
                    ax.text(ca_pixels[0][mm]+5,ca_pixels[1][mm],f'{mm}_{i}',c='k',fontsize=25) #shadow
                    ax.text(ca_pixels[0][mm]+4,ca_pixels[1][mm],f'{mm}_{i}',c=colors[i%10],fontsize=25)
            plt.title('Black ellipses show sources in final \nDR1 value-added catalogue.' \
                 '\n White \'+\'-markers show sources found in noisified \nimage by PyBDSF.')
            plt.show()
        if debug:
            print("We found the following matches:", unique_matches)

    if not unique_matches:
        print("No matches found in original sourcelist and sourcelist of augmented cutout.")
        stats = stats.append({'Cat':f'{gaussian_blur:.2f} sigma blur, {noise} sigma noise',
                          'Detected sources':len(augmented_cat),
                          'Detected flux (mJy)':augmented_cat.Total_flux.sum()*1000,
                          'Linked sources':0,
                          'Linked flux (mJy)':0,
                          'Linked flux fraction':0,
                          'Unassociated sources':int(len(augmented_cat)),
                         'Unassociated flux (mJy)':augmented_cat.Total_flux.sum()*1000,
                         'Unassociated flux fraction':augmented_cat.Total_flux.sum()*1000/original_flux},ignore_index=True)
    elif not os.path.exists(augmented_cat_fits_path):
        stats = stats.append({'Cat':f'{gaussian_blur:.2f} sigma blur, {noise} sigma noise',
                          'Detected sources':0,
                          'Detected flux (mJy)':0,
                          'Linked sources':0,
                          'Linked flux (mJy)':0,
                          'Linked flux fraction':0,
                          'Unassociated sources':0,
                         'Unassociated flux (mJy)':0,
                         'Unassociated flux fraction':0},ignore_index=True)

    else:
        comp_keys = ['Source_Name', 'RA', 'E_RA', 'DEC', 'E_DEC', 'Peak_flux',
               'E_Peak_flux', 'Total_flux', 'E_Total_flux', 'Maj', 'E_Maj', 'Min',
               'E_Min', 'DC_Maj', 'E_DC_Maj', 'DC_Min', 'E_DC_Min', 'PA', 'E_PA',
               'DC_PA', 'E_DC_PA', 'Isl_rms', 'S_Code', 'Mosaic_ID',
               'Component_Name']

        # Get unique combined names
        # comp_names = get_comps(predicted_comp_cat)

        # Get component lists for each predicted source
        clists = [augmented_cat.iloc[m] for m in new_matches[np.argsort(len_matches_ranked)]]
        clists = [l for l in clists if not l.empty]


        # Iterate over clists
        # Modelled after https://github.com/mhardcastle/lotss-catalogue/blob/master/catalogue_create/process_lgz.py
        # Got it working for everything but size
        keys = ['Source_id','RA','E_RA','DEC','E_DEC','Peak_flux','E_Peak_flux','Total_flux','E_Total_flux',
                'Isl_rms','S_Code']
        #predicted_source_cat = pd.DataFrame(columns=keys, data={})
        data = []
        compdata = []
        for irow, clist in enumerate(clists):
            #clist = clist_series.to_dict('records')
            #print('irow',type(irow),np.shape(irow), irow, 'clist',type(clist), np.shape(clist), clist)
            ms=Make_Shape(clist)
            r = {}
            r['Predicted_Size']=ms.length()
            r['Predicted_Width']=ms.width()
            r['Predicted_PA']=ms.pa()
            # WHETHER THE SOURCE HAS CHANGED OR NOT,
            # recreate all of the source's radio info from its component list
            # this ensures consistency

            r['Predicted_Assoc']=len(clist)
            if len(clist)==1:
                # only one component, so we can use its properties
                c=clist.iloc[0]
                for key in keys:
                    r[key]=c[key]
                rc = deepcopy(c)
                sname = sourcename(rc.RA, rc.DEC)
                compdata.append(rc)
            else:
                tfluxsum=clist['Total_flux'].sum()
                ra=np.sum(clist['RA']*clist['Total_flux'])/tfluxsum
                dec=np.sum(clist['DEC']*clist['Total_flux'])/tfluxsum
                sname=sourcename(ra,dec)
                for icomp, comp in clist.iterrows():
                    rc = deepcopy(comp)
        #             rc['Component_Name'] = rc['Source_Name']
                    rc['Source_Name'] = sname
                    compdata.append(rc)
                r['Source_id']= clist['Source_id'].values
                r['RA']=ra
                r['DEC']=dec
                r['Source_Name']=sname
                r['E_RA']=np.sqrt(np.mean(np.power(clist['E_RA'],2)))
                r['E_DEC']=np.sqrt(np.mean(np.power(clist['E_DEC'],2)))
                r['Source_Name']=sname
                r['Total_flux']=tfluxsum
                r['E_Total_flux']=np.sqrt(np.sum(np.power(clist['E_Total_flux'],2)))
                maxpk=clist['Peak_flux'].idxmax()
                r['Peak_flux']=clist.loc[maxpk]['Peak_flux']
                r['E_Peak_flux']=clist.loc[maxpk]['E_Peak_flux']
                r['S_Code']='M'
                r['Isl_rms']=np.mean(clist['Isl_rms'])
            data.append(r)
        retrieved_component_cat = pd.DataFrame(columns=comp_keys, data=compdata).reset_index(drop=True)
        retrieved_source_cat = pd.DataFrame(columns=keys+['Predicted_Size','Predicted_Width','Predicted_PA',
                                                      'Predicted_Assoc'], data=data)
        # save_cat(predicted_source_cat, cfg.OUTPUT_DIR, 'LoTSS_predicted_v0_merge',
        #          keys=keys+['Predicted_Size','Predicted_Width','Predicted_PA','Predicted_Assoc'] )
        # save_cat(source_component_cat, cfg.OUTPUT_DIR, 'LoTSS_predicted_v0.comp', keys=comp_keys)

        #########################################
        """
        Catalogue contents explained.
        cutout_cat: LoTSS DR1 value added catalogue filtered for this single cutout.
        augmented_cat: PyBDSF ran over the cutout after adding noise to the cutout
        retrieved_component_cat: augmented_cat minus the sources that could not be linked to cutout_cat
        unassoc_cat: augmented_cat minus linked sources (complement of retrieved_component_cat)

        retrieved_source_cat: retrieved_component_cat sources combined into single sources linked to cutout_cat
        """

        # Write up the statistics/results
        ii = [i for i in augmented_cat.index if not i in unique_matches]
        unassoc_cat = augmented_cat.iloc[ii]
        #print('unique matches', unique_matches, "augmented cat", augmented_cat)
        #print('unassoc_cat', unassoc_cat, "total", unassoc_cat.Total_flux.sum()*1000)
        original_flux = cutout_cat.Total_flux.sum()
        stats = stats.append({'Cat':f'{gaussian_blur:.2f} sigma blur, {noise} sigma noise',
                              'Detected sources':len(augmented_cat),
                              'Detected flux (mJy)':augmented_cat.Total_flux.sum()*1000,
                              'Linked sources':len(retrieved_component_cat),
                              'Linked flux (mJy)':retrieved_source_cat.Total_flux.sum()*1000,
                              'Linked flux fraction':retrieved_source_cat.Total_flux.sum()*1000/original_flux,
                              'Unassociated sources':int(len(unassoc_cat)),
                             'Unassociated flux (mJy)':unassoc_cat.Total_flux.sum()*1000,
                             'Unassociated flux fraction':unassoc_cat.Total_flux.sum()*1000/original_flux},ignore_index=True)

stats['Detected sources'] = stats['Detected sources'].astype(int)
stats['Linked sources'] = stats['Linked sources'].astype(int)
stats['Unassociated sources'] = stats['Unassociated sources'].astype(int)
print(stats)
