import torch 
import ee
import geemap
import rasterio
import rioxarray as rxr
import numpy as np
import os
from joblib import Parallel, delayed
import glob
import model_resnet_50
import model_max_vision_transformer
from tqdm import tqdm
import itertools
import ssl
import gc   
import model_swin_transformer
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Patch
ssl._create_default_https_context = ssl._create_unverified_context


class DeepSAR():
    def __init__(self,path,geometry,date):
        self.path=path
        self.geometry=geometry
        self.date=date
        
    def Initialize(self):
    # Initialize the Earth Engine module.
        try:
            ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
        except Exception as e:
            ee.Authenticate()
            ee.Initialize()
        print('Earth Engine initialized.')

    def gpu_check(self):
        gpus = torch.cuda.device_count()
        if gpus>0:
            print(gpus, "Physical GPUs,")
            for i in range(gpus):
                print("GPU", i, torch.cuda.get_device_name(i))
            return('cuda')
        else:
            print('GPU not found')
        return('cpu')

    def setup(self):
        self.Initialize()
        return self.gpu_check()

    def emptyDirectory(self,dirPath, ext = "*", verbose = True):
        allFiles = glob.glob(dirPath + "/" + ext)
        if verbose:
            print(str(len(allFiles)) + " files found in " + dirPath + " --> Will be deleted.")
        for f in allFiles:
            os.remove(f)

    def createDirectory(self,dirPath=os.path.expanduser('~/Downloads/SAR_flood'), emptyExistingFiles = False, verbose = True):
        if not os.path.isdir(dirPath):
            os.makedirs(dirPath)
            if verbose:
                print("Folder not found!!!   " + dirPath + " created.")
                print('------Folder Creation Done!!!--------')
        else:
            print('%s --> Folder exists!!!'%dirPath)
            print('------------------Using existing folder!!!-----------------')
            if emptyExistingFiles:
                self.emptyDirectory(dirPath, verbose = verbose)
        return(dirPath)
    
    def download_gee_image(self,band,file_name,geometry):
        geemap.download_ee_image(band,file_name,crs="EPSG:4326",region=geometry, scale=10)

    def gee_files(self,download=True):
        self.Initialize()
        ee_date=ee.Date(self.date)
        ee_geometry =ee.Geometry.Rectangle(self.geometry)
        name='gee_files'+'_'+'_'.join([str(elem) for elem in self.geometry])
        file_path=self.createDirectory(os.path.join(self.path,name))

        jrc = ee.Image('JRC/GSW1_4/GlobalSurfaceWater').select('seasonality')
        dem=ee.Image('NASA/NASADEM_HGT/001').select('elevation')
        slope = ee.Terrain.slope(dem)
        hand= ee.ImageCollection("users/gena/global-hand/hand-100").mosaic().rename('hand')


        s1_file_name=os.path.join(file_path,'s1_'+'_'.join([str(elem) for elem in self.geometry])+'.tif')
        dem_file_name=os.path.join(file_path,'dem_'+'_'.join([str(elem) for elem in self.geometry])+'.tif')
        slope_file_name=os.path.join(file_path,'slope_'+'_'.join([str(elem) for elem in self.geometry])+'.tif')
        jrc_file_name=os.path.join(file_path,'jrc_'+'_'.join([str(elem) for elem in self.geometry])+'.tif')
        hand_file_name=os.path.join(file_path,'hand_'+'_'.join([str(elem) for elem in self.geometry])+'.tif')

        sen1= ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(ee_geometry).filterDate(ee_date,ee_date.advance(10,'days')).mosaic().clip(ee_geometry).select(['VV','VH'])
        list=[sen1,dem,slope,jrc,hand]
        name=[s1_file_name,dem_file_name,slope_file_name,jrc_file_name,hand_file_name]
        if download:
            print("--------------Files download getting ready ---------------------")
            Parallel(n_jobs=5,backend='threading')(delayed(self.download_gee_image)(list[i],name[i],ee_geometry) for i in range(len(list)))
            print('--------------Files Download Done!!!------------------------')
        return(s1_file_name,dem_file_name,slope_file_name,jrc_file_name,hand_file_name)
    
    
    def create_data(self,download=True):
        features = []  
        images = []
        s1,dem,slope,jrc,hand=self.gee_files(download)
        with rasterio.open(s1) as img:
            ar=np.float32(np.clip(img.read(1), -35, 0)/-35)
            ar[np.isnan(ar)] = 0
            images.append(ar)
            ar=np.float32(np.clip(img.read(2), -42, -5)/-42)
            ar[np.isnan(ar)] = 0
            images.append(ar)
        with rasterio.open(jrc) as img:
            ar=np.array(np.float32(np.clip(img.read(1), 0, 12)/12))
            ar[np.isnan(ar)] = 0
            images.append(ar)
        with rasterio.open(dem) as img:
            ar=np.array(np.float32(np.clip(img.read(1), 0, 250)/250))
            ar[np.isnan(ar)] = 0
            images.append(ar)
        with rasterio.open(slope) as img:
            ar=np.array(np.float32(np.clip(img.read(1), 0, 5)/5))
            ar[np.isnan(ar)] = 0
            images.append(ar)
        with rasterio.open(hand) as img:
            ar=np.array(np.float32(np.clip(img.read(1), 0, 1000)/1000))
            ar[np.isnan(ar)] = 0
            images.append(ar)
        features.append(np.stack(images, axis=-1)) 
        dl_files=np.array(features)
        return dl_files[0],s1
    
    def sliding_window(self,top, step, window_size):
        """ Slide a window_shape window across the image with a stride of step """
        for x in range(0, top.shape[0], step):
            if x + window_size[0] > top.shape[0]:
                x = top.shape[0] - window_size[0]
            for y in range(0, top.shape[1], step):
                if y + window_size[1] > top.shape[1]:
                    y = top.shape[1] - window_size[1]
                yield x, y, window_size[0], window_size[1]
        
            
    def count_sliding_window(self,top, step=400, window_size=(512,512)):
        """ Count the number of windows in an image """
        c = 0
        for x in range(0, top.shape[0], step):
            if x + window_size[0] > top.shape[0]:
                x = top.shape[0] - window_size[0]
            for y in range(0, top.shape[1], step):
                if y + window_size[1] > top.shape[1]:
                    y = top.shape[1] - window_size[1]
                c += 1
        return c

    def grouper(self,n, iterable):
        """ Browse an iterator by chunk of n elements """
        it = iter(iterable)
        while True:
            chunk = tuple(itertools.islice(it, n))
            if not chunk:
                return
            yield chunk




    def load_models(self):
        model_weights_swin='./weights/unet_swinv2_base_window8_256.pth'
        model_swin= model_swin_transformer.Unet()
        model_swin.load_state_dict(torch.load(model_weights_swin))

        model_weights_maxvit='./weights/unet_maxvit_small_tf_224.pth'
        model_maxvit= model_max_vision_transformer.Unet()
        model_maxvit.load_state_dict(torch.load(model_weights_maxvit))


        model_weights_resnet='./weights/unetplusplus_resnet50.pth'
        model_resnet= model_resnet_50.UnetPlusPlus()
        model_resnet.load_state_dict(torch.load(model_weights_resnet))
        return [model_swin,model_maxvit,model_resnet]
    



    def create_model_probability(self,model_name,tiff=True,download=True,stride=400,batch_size=12):
        device=self.setup()
        if model_name not in ['swin','maxvit','resnet']:
            raise ValueError('Model name should be either swin, maxvit or resnet')
        if model_name=='swin':
            model=self.load_models()[0]
        elif model_name=='maxvit':
            model=self.load_models()[1]
        else:
            model=self.load_models()[2]

        window_size=(512,512)
        all_preds = []
        img,s1=self.create_data(download)
    # Switch the network to inference mode
        model.to(device)
        model.eval()
        pred = np.zeros(img.shape[:2] + (2,))
        count = np.zeros(img.shape[:2])

        total = self.count_sliding_window(img, step=stride, window_size=window_size) // batch_size
        for i, coords in enumerate(self.grouper(batch_size, self.sliding_window(img, step=stride, window_size=window_size))):
            # Build the tensor
            image_patches = torch.Tensor(np.array([np.copy(img[x:x+w, y:y+h]).transpose(2,0,1) for x,y,w,h in coords])).to(device)

            # Do the inference
            with torch.inference_mode():
                outs = torch.softmax(model(image_patches), dim=1)
                outs = outs.data.cpu().numpy()
            
            # Fill in the results array
            for out, (x, y, w, h) in zip(outs, coords):
                out = out.transpose((1,2,0))
                pred[x:x+w, y:y+h]+= out
                count[x:x+w, y:y+h]+=1
            del(outs)

        probability=pred[:,:,1]
        probability=probability/count
        del model
        del image_patches
        gc.collect()
        torch.cuda.empty_cache()
    
        if tiff:
            name=self.path+'/flood_probability_'+self.date+'_'.join([str(elem) for elem in self.geometry])+'.tif'
            img=rasterio.open(s1)
            dict=img.meta
            dataset = rasterio.open(
                            name, "w", 
                            driver = dict['driver'],
                            height = dict['height'],
                            width = dict['width'],
                            count = 1,
                            nodata =dict['nodata'],
                            dtype = dict['dtype'],
                            crs = dict['crs'],
                            transform = dict['transform'])
            dataset.write(probability,1)
            dataset.set_band_description(1, 'flood_probability')
            dataset.close()
            return probability,dataset.name
        return probability
    
    def create_multi_model_probability(self,tiff=True,individual_prob=False,download=True,stride=400,batch_size=12):
        device=self.setup()
        models=self.load_models()
        window_size=(512,512)
        all_preds = []
        img,s1=self.create_data(download)
        for i in range(len(models)):
            print(f'Running model {i+1}/{len(models)}')
            pred = np.zeros(img.shape[:2] + (2,))
            model=models[i]
            model.to(device)
            model.eval()
            count = np.zeros(img.shape[:2])

            total = self.count_sliding_window(img, step=stride, window_size=window_size) // batch_size
            for i, coords in enumerate(tqdm((self.grouper(batch_size, self.sliding_window(img, step=stride, window_size=window_size))))):
                # Build the tensor
                image_patches = torch.Tensor(np.array([np.copy(img[x:x+w, y:y+h]).transpose(2,0,1) for x,y,w,h in coords])).to(device)

                # Do the inference
                with torch.inference_mode():
                    outs = torch.softmax(model(image_patches), dim=1)
                    outs = outs.data.cpu().numpy()
                
                # Fill in the results array
                for out, (x, y, w, h) in zip(outs, coords):
                    out = out.transpose((1,2,0))
                    pred[x:x+w, y:y+h]+= out
                    count[x:x+w, y:y+h]+=1
                del(outs)

            probability=pred[:,:,1]
            probability=probability/count
            
            all_preds.append(probability)
            del model
            del image_patches
            gc.collect()
            torch.cuda.empty_cache()
        all_probs=np.array(all_preds)
        mean=np.mean(all_probs,axis=0) 
        label=(mean>0.5)*1
        uncertainty=(label==1)*2-(2/3*(mean))
        uncertainty=(label==0)*2/3*(mean)
        if tiff:
            name=self.path+'/flood_probability_'+self.date+'_'+'_'.join([str(elem) for elem in self.geometry])+'.tif'
            img=rasterio.open(s1)
            dict=img.meta
            if individual_prob:
                dataset = rasterio.open(
                                name, "w", 
                                driver = dict['driver'],
                                height = dict['height'],
                                width = dict['width'],
                                count = 5,
                                nodata =dict['nodata'],
                                dtype = dict['dtype'],
                                crs = dict['crs'],
                                transform = dict['transform'])
                dataset.write(label,1)
                dataset.set_band_description(1, 'flood_label')
                dataset.write(uncertainty,2)
                dataset.set_band_description(2, 'uncertainty')
                dataset.write(all_probs[0],3)
                dataset.set_band_description(3, 'flood_prob_1')
                dataset.write(all_probs[1],4)
                dataset.set_band_description(4, 'flood_prob_2')
                dataset.write(all_probs[2],5)
                dataset.set_band_description(5, 'flood_prob_3')
                dataset.close()
                return label,uncertainty,dataset.name  
            else:
                dataset = rasterio.open(
                                name, "w", 
                                driver = dict['driver'],
                                height = dict['height'],
                                width = dict['width'],
                                count = 2,
                                nodata =dict['nodata'],
                                dtype = dict['dtype'],
                                crs = dict['crs'],
                                transform = dict['transform'])
                dataset.write(label,1)
                dataset.set_band_description(1, 'flood_label')
                dataset.write(uncertainty,2)
                dataset.set_band_description(2, 'uncertainty')
                dataset.close()
                return label,uncertainty,dataset.name           

    def input_visualization(self, download=False):
        input=self.gee_files(download=download)

        water_cm= LinearSegmentedColormap.from_list( 'cm', ['#FFFFFF','#0000FF'], N=2)

        file=rxr.open_rasterio(input[0])
        jrc=rxr.open_rasterio(input[-2])
        dem=rxr.open_rasterio(input[1])
        slope=rxr.open_rasterio(input[2])
        hand=rxr.open_rasterio(input[-1])

        fig,ax=plt.subplots(1,6,figsize=(24,8),)
        num_colors = 20
        cmap = plt.get_cmap('Greys_r', num_colors)

        im=ax[0].imshow(file[0],cmap=cmap,vmin=-22,vmax=-5)
        # ax[0].set_title('SAR Band - VV')
        divider = make_axes_locatable(ax[0])
        cax = divider.new_vertical(size = "4%",pad = 0.2,pack_start = True)
        # Create colorbar
        ax[0].figure.add_axes(cax)
        cbar = ax[0].figure.colorbar(im, cax = cax,orientation = 'horizontal')
        ax[0].set_xticks([])
        ax[0].set_yticks([])


        im=ax[1].imshow(file[1],cmap=cmap,vmin=-30,vmax=0)
        # ax[1].set_title('SAR band-VH')
        divider = make_axes_locatable(ax[1])
        cax = divider.new_vertical(size = "4%",pad = 0.2,pack_start = True)
        # Create colorbar
        ax[1].figure.add_axes(cax)
        cbar = ax[1].figure.colorbar(im, cax,orientation = 'horizontal')
        ax[1].set_yticks([])
        ax[1].set_xticks([])


        im=ax[2].imshow(dem[0],cmap='terrain',vmin=0,vmax=40)
        # ax[2].set_title('DEM')
        divider = make_axes_locatable(ax[2])
        cax = divider.new_vertical(size = "4%",pad = 0.2,pack_start = True)
        # Create colorbar
        ax[2].figure.add_axes(cax)
        cbar = ax[2].figure.colorbar(im, cax,orientation = 'horizontal')
        ax[2].set_yticks([])
        ax[2].set_xticks([])

        im=ax[3].imshow(slope[0],vmin=0,vmax=10)
        # ax[3].set_title('Slope')
        divider = make_axes_locatable(ax[3])
        cax = divider.new_vertical(size = "4%",pad = 0.2,pack_start = True)
        # Create colorbar
        ax[3].figure.add_axes(cax)
        cbar = ax[3].figure.colorbar(im, cax,orientation = 'horizontal')
        ax[3].set_yticks([])
        ax[3].set_xticks([])


        im=ax[4].imshow(jrc[0],cmap=water_cm)
        divider = make_axes_locatable(ax[4])
        cax = divider.new_vertical(size = "4%",pad = 0.2,pack_start = True)
        # ax[4].set_title('JRC Permanent Water')
        legend_labels ={'Water':'#0000FF'}
        patches = [Patch(color=color, label=label)for label, color in legend_labels.items()]
        ax[4].legend(handles=patches,bbox_to_anchor=(0.8, -0.00), prop={'size':14},title_fontsize= 14,facecolor='white')
        ax[4].set_yticks([])
        ax[4].set_xticks([])

        im=ax[5].imshow(hand[0],vmin=0,vmax=20)
        divider = make_axes_locatable(ax[5])
        cax = divider.new_vertical(size = "4%",pad = 0.2,pack_start = True)
        ax[5].figure.add_axes(cax)
        cbar = ax[5].figure.colorbar(im, cax,orientation = 'horizontal')
        ax[5].set_yticks([])
        ax[5].set_xticks([])
        plt.grid(False)
        fig.tight_layout(rect=[0, 0.01, 1, 1.2])
        plt.show()

    def plot_output(self):
        name=self.path+'/flood_probability_'+self.date+'_'+'_'.join([str(elem) for elem in self.geometry])+'.tif'
        image=rxr.open_rasterio(name)
        water_cm= LinearSegmentedColormap.from_list( 'cm', ['#FFFFFF','#0000FF'], N=2)
        uncertainty_cm= LinearSegmentedColormap.from_list( 'cm', ['#FFFFFF','#FF0000'], N=10)
        if len(image.band.values)==5:
            fig,ax=plt.subplots(1,5,figsize=(24,8))
            l=[2,3,4,0]

            for j in range(4):
                im=image[l[j]]
                ax[j].imshow(im,vmin=0,vmax=1,cmap=water_cm)
                divider = make_axes_locatable(ax[j])
                cax = divider.new_vertical(size = "4%",pad = 0.2,pack_start = True)
                legend_labels ={'Water':'#0000FF'}
                patches = [Patch(color=color, label=label)for label, color in legend_labels.items()]
                ax[j].legend(handles=patches,bbox_to_anchor=(0.6, 0), prop={'size':10},title_fontsize= 10,facecolor='white')
                ax[j].set_yticks([])
                ax[j].set_xticks([])

            im=image[1]
            im=ax[4].imshow(im,cmap=uncertainty_cm,vmin=0,vmax=1)
            ax[4].set_title('Uncertainty',fontsize=10)
            divider = make_axes_locatable(ax[4])
            cax = divider.new_vertical(size = "4%",pad = 0.2,pack_start = True)
            # Create colorbar
            ax[4].figure.add_axes(cax)
            cbar = ax[4].figure.colorbar(im, cax,orientation = 'horizontal')
            ax[4].set_yticks([])
            ax[4].set_xticks([])
            plt.grid(False)
            fig.tight_layout(rect=[0, 0.01, 1, 1.2])
            plt.show()
        if len(image.band.values)==2:
            fig,ax=plt.subplots(1,2,figsize=(24,8))
            im=image[0]
            ax[0].imshow(im,vmin=0,vmax=1,cmap=water_cm)
            divider = make_axes_locatable(ax[0])
            cax = divider.new_vertical(size = "4%",pad = 0.2,pack_start = True)
            legend_labels ={'Water':'#0000FF'}
            patches = [Patch(color=color, label=label)for label, color in legend_labels.items()]
            ax[0].legend(handles=patches,bbox_to_anchor=(0.6, 0), prop={'size':10},title_fontsize= 10,facecolor='white')
            ax[0].set_yticks([])
            ax[0].set_xticks([])

            im=image[1]
            im=ax[1].imshow(im,cmap=uncertainty_cm,vmin=0,vmax=1)
            ax[1].set_title('Uncertainty',fontsize=10)
            divider = make_axes_locatable(ax[1])
            cax = divider.new_vertical(size = "4%",pad = 0.2,pack_start = True)
            # Create colorbar
            ax[1].figure.add_axes(cax)
            cbar = ax[1].figure.colorbar(im, cax,orientation = 'horizontal')
            ax[1].set_yticks([])
            ax[1].set_xticks([])


            plt.grid(False)
            fig.tight_layout(rect=[0, 0.01, 1, 1.2])
            plt.show()  