import matplotlib.pyplot as plt
import numpy as np
import csv
import os
import glob
import math
import pylab
import matplotlib.font_manager as font_manager

from scipy.stats import f_oneway
from scikit_posthocs import posthoc_ttest

class FileManagement:
    
    # Create a folder
    # @input dir_name String The directory's name
    def create_folder(self,dir_name):
        if type(dir_name)!=str or dir_name=="":
            raise Exception("[plot_results.py] Wrong input");
        dirname = os.path.dirname(__file__);
        dir_to_create = os.path.join(dirname,dir_name);

        # If directories already exist, pass
        try:
            os.makedirs(dir_to_create);
        except FileExistsError:
            pass;
        return 0;

    # Get the paths of all the files having a specific extension
    # @input folder String. Path to the folder containing all the csv results files
    # @input extension  String. Extension of the files we want to retrieve from the directory
    # @return A list containing all the absolute paths of the files having the desired extension
    def get_allcsvfiles(self,folder,extension):
        if type(folder)!=str or folder=='' or type(extension)!=str or extension=='':
            raise Exception("[plot_results.py] Wrong input");
        folder = folder + '*.' + extension;
        files = glob.glob(folder);
        if len(files)<=0 or type(files)!=list:
            raise Exception("[plot_results.py] Folder is empty");
        return files;

class StatisticalAnalysis:

    def __init__(self):
        self.file_mangmt = FileManagement();

    # Load the csv file containing all results and convert the csv into a 2D numpy array
    # @input filename String. filename The file name of the results file
    # @return 2D numpy array
    def convert_csv2mat(self,filename):
        if type(filename)!=str or filename=='':
            raise Exception("[plot_results.py] Wrong input");
        return np.loadtxt(open(filename, "rb"), delimiter=",", skiprows=0); 

    # Convert all the csv files in the folder to a numpy array
    # @input filenames_list List. It is a list of all the absolute paths of the csv files
    # @return List of the corresponding numpy arrays
    def convert_allcsv2mat(self,filenames_list):
        if type(filenames_list)!=list or len(filenames_list)<=0:
            raise Exception("[plot_results.py] Wrong input");
        resultsArray_list = [];
        for filename in filenames_list:
            resultsArray_list.append(self.convert_csv2mat(filename));
        if len(resultsArray_list)<=0 or type(resultsArray_list)!=list:
            raise Exception("[plot_results.py] The list is empty");
        return resultsArray_list;

    # Compute the confidence interval with the lower and upper bound errors
    # @input input Numpy array containing the data for the computation of the confidence interval. It is a column matrix.
    # @input alpha Float. Confidence level. Can only be between 0 and 1.
    # @return Numpy array. First element is the lower bound. Second element is the upper bound. This is the confidence interval.
    def confidence_interval(self,input,alpha):
        row_input,col_input = input.shape;
        if type(input)!=np.ndarray or type(alpha)!=float or alpha<0 or row_input<=0 or col_input<=0:
            raise Exception("[plot_results.py] Wrong input");
        results = np.zeros((2,col_input));# Matrix containing the lower and upper bounds.
        lower_bound = np.mean(input,axis=0) - alpha*((np.std(input,axis=0)/(math.sqrt(row_input))));
        upper_bound = np.mean(input,axis=0) + alpha*((np.std(input,axis=0)/(math.sqrt(row_input))));
        results[0,:] = lower_bound;results[1,:] = upper_bound;
        return results;

    # Compute the standard error
    # @input input Numpy array containing the data for the compuutation of the standard error. It is a column matrix.
    # @return Numpy array. Each column corresponds to the standard error for each data input.
    def standard_error(self,input):
        row_input,col_input = input.shape;
        if type(input)!=np.ndarray or row_input<=0 or col_input<=0:
            raise Exception("[plot_results.py] Wrong input");
        results = results = np.zeros((1,col_input));# Matrix containing the standard errors.
        std_error = (np.std(input,axis=0))/(math.sqrt(row_input));
        std_error = std_error.reshape((1,-1));
        return std_error;

    # Compute the means and standard deviations of all the arrays in an list of arrays
    # @input arraysList List. It is a list of arrays. [A0,....,AN]
    # @return A numpy array. Each row of the array contains the mean of each column in [A0,...,AN]
    # @return A numpy array. Reach row of the array contains the standard deviation of each columns in [A0,...,AN]
    def compute_statistics(self,arraysList):
        if type(arraysList)!=list or len(arraysList)<=0:
            raise Exception("[plot_results.py] Wrong input");
        rows,cols = arraysList[0].shape;
        means_mat = np.zeros((len(arraysList),cols));
        stds_mat = np.zeros((len(arraysList),cols));
        stderr_list = [];# List containing all the standard errors for each column

        k = 0;
        for mat in arraysList:
            means_mat[k,:] = np.mean(mat,axis=0);
            stds_mat[k,:] = np.std(mat,axis=0);
            stderr_mat = self.standard_error(mat);# Compute the confidence intervals
            stderr_list.append(stderr_mat);
            k = k + 1;
        if len(stderr_list)!=len(arraysList) or means_mat.shape[0]!=len(arraysList) or stds_mat.shape[0]!=len(arraysList) or means_mat.shape[1]!=cols or stds_mat.shape[1]!=cols:
            raise Exception("[plot_results.py] Wrong computations.");
        return means_mat,stds_mat,stderr_list;

    # Get the p-values for each array. The one-way ANOVA is applied
    # @input arraysList List. It is a list of arrays. [A0,....,AN]
    # @return A numpy array. Reach row of the array contains the p-value of each sample in [A0,...,AN]
    def get_arraysList_pValues(self,arraysList):
        if type(arraysList)!=list or len(arraysList)<=0:
            raise Exception("[plot_results.py] Wrong input");
        p_values_array = np.zeros((len(arraysList),1));
        k = 0;
        for mat in arraysList:
            col0 = mat[:,0];
            col1 = mat[:,1];
            col2 = mat[:,2];
            col3 = mat[:,3];
            F, p_value = f_oneway(col0,col1,col2,col3,axis=0);
            p_values_array[k] = p_value;
            k = k + 1;
    
        if p_values_array.shape[0]!=len(arraysList) or p_values_array.shape[1]!=1:
            raise Exception("[plot_results.py] Wrong computations");
        return p_values_array;

    # Get the results of the post-hoc analysis after the one-way ANOVA has been done
    # @input arraysList List of arrays
    # @return List of Numpy arrays
    def get_arraysList_postHoc(self,arraysList):
        if type(arraysList)!=list or len(arraysList)<=0:
            raise Exception("[plot_results.py] Wrong input");
        p_values_list = [];
        for mat in arraysList:
            mat_list = np.split(mat,arraysList[0].shape[1],axis=1);
            p_values = posthoc_ttest(mat_list);
            p_values = p_values.to_numpy();
            p_values_list.append(p_values);
        if len(p_values_list)!=len(arraysList):
            raise Exception("[plot_results.py] Wrong computation");
        return p_values_list;

    # Run all the statistical analysis for the computational complexities
    # @input folder String.The absolute folder pathname where all the results are saved by the algorithm
    # @return A list containing all the statistical results in the following order: Matrices with all values,means,standard deviations,p_values with ANOVA,p_values with t_test from post-hoc analysis
    def run_analysis_complexities(self,folder):
        if type(folder)!=str or folder=='':
            raise Exception("[plot_results.py] Wrong input.");
        filenames_list = self.file_mangmt.get_allcsvfiles(folder,'csv');
        resultsArray_list = self.convert_allcsv2mat(filenames_list);
        output = [resultsArray_list,filenames_list];
        return output;

    # Run all the statistical analysis
    # @input folder String.The absolute folder pathname where all the results are saved by the algorithm
    # @return A list containing all the statistical results in the following order: Matrices with all values,means,standard deviations,p_values with ANOVA,p_values with t_test from post-hoc analysis
    def run_analysis(self,folder):
        if type(folder)!=str or folder=='':
            raise Exception("[plot_results.py] Wrong input.");
        filenames_list = self.file_mangmt.get_allcsvfiles(folder,'csv');
        resultsArray_list = self.convert_allcsv2mat(filenames_list);
        means_mat,stds_mat,std_list = self.compute_statistics(resultsArray_list);
        p_values_array = self.get_arraysList_pValues(resultsArray_list);
        post_hoc_values = self.get_arraysList_postHoc(resultsArray_list);
        output = [resultsArray_list,means_mat,stds_mat,std_list,p_values_array,post_hoc_values,filenames_list,resultsArray_list];
        if len(output)<=0:
            raise Exception("[plot_results,py] The list containing the results is empty.");
        return output;

class Graphics:

    def __init__(self):
        self.file_mangmt = FileManagement();

    # Save the figure's legend in the project's root directory
    # @input legend The legend to save.
    # @input filename The legend figure's filename.
    # @input expand The expansion.
    def export_legend(self,legend,filename="legend.png",expand=[-5,-5,5,5]):
        fig  = legend.figure;
        fig.canvas.draw();
        bbox  = legend.get_window_extent();
        bbox = bbox.from_extents(*(bbox.extents + np.array(expand)));
        bbox = bbox.transformed(fig.dpi_scale_trans.inverted());
        fig.savefig(filename, dpi=600, bbox_inches=bbox,transparent=True);
        return 0;

    # Save a figure
    # @input fig The figure to save
    # @input filename The filename under which the picture is saved
    def save_fig(self,fig,filename):
        if type(filename)!=str or filename=="":
            raise Exception("[plot_results.py] Wrong input");
        fig.savefig(filename,format='png',dpi=600,transparent=True);
        return 0;
        
    # Plot a bar plot using the input data
    # @input labels_ticks List. Labels that goes under each bar in the bar plot
    # @input data Numpy array. Data to plot
    # @input stderr_mat Numpy array. Standard errors
    # @input xlabel String. Label for the plot's x axes
    # @input ylabel String. Label for the plot's y axes
    # @input title String. Plot's title
    # @input colours_list List of colours
    # @input x Set the separation space between the bars
    # @input width The width of the bars
    # @return The figure that has been plotted
    def bar_plot(self,data,stderr_mat,labels_ticks,xlabel,ylabel,title,colours_list,x,width):
        if type(stderr_mat)!=np.ndarray or type(labels_ticks)!=list or type(data)!=np.ndarray or type(xlabel)!=str or type(ylabel)!=str or type(title)!=str:
            raise Exception("[plot_results.py] Wrong input");
        if type(colours_list)!=list or len(colours_list)!=data.shape[1] or type(x)!=list or len(x)!=data.shape[1] or width<=0:
            raise Exception("[plot_results,py] Wrong input");
        if len(labels_ticks)!=data.shape[1] or stderr_mat.shape[0]!=1 or stderr_mat.shape[1]!=data.shape[1]:
            raise Exception("[plot_results.py] The number of elements in labels_ticks must be the same as the number of columns in the data");
        fig = plt.figure(figsize=(10,5));
        ax = plt.axes();

        # creating the bar plot. We must reshape the array, so it becomes a 1D vector because bar() does not accept 2D-arrays
        #width = 0.05;  # the width of the bars
        #x = [0,width,2*width,3*width];# The graphs are closed to each other
        data = data.reshape(-1).tolist();
        stderr_mat = stderr_mat.reshape(-1).tolist();
        #colours_list = ["#CAA393","#8DB6D5","#C0A9CF","#98C5B2"];
        plt.grid(axis='y',**{"color":'white',"zorder":1});

        for pst,y,stderr,colour,label_tick in zip(x,data,stderr_mat,colours_list,labels_ticks):
            plt.bar(pst,y,width,yerr=stderr,label=label_tick,**{"zorder":2,"color":[colour]}); 

        font = font_manager.FontProperties(family='Arial',style='normal',size=25);
        legend = plt.legend(fontsize=15,prop=font,facecolor="#EBECF3");
        self.export_legend(legend);

        # Modifications to the graph. Remove x-axis and y-axis small ticks. The x-axis labels are removed.
        # A grid along the x-axis is added. The background colour is changed
        plt.tick_params(axis=u'both',which=u'both',length=0,labelbottom=False);
        plt.yticks(**{"fontname":"Arial","fontsize":25});
        ax.set_facecolor("#EBECF3");
        ax.get_legend().remove();

        # Showing the graph in full-screen mode
        plt.xlabel(xlabel, **{"fontname":"Arial","fontsize":25});
        plt.ylabel(ylabel, **{"fontname":"Arial","fontsize":25});
        plt.title(title, **{"fontname":"Arial","fontsize":25});
        #plt.show();
        return fig;

    # Plot Tukey boxplots for the data
    def tukey_box_plot(self,data,xlabel,ylabel,title,colours_list):
        if type(data)!=np.ndarray or type(xlabel)!=str or type(ylabel)!=str or type(title)!=str:
            raise Exception("[plot_results.py] Wrong input");
        if data.shape[0]<=0 or data.shape[1]<=0 or len(colours_list)!=data.shape[1] or type(colours_list)!=list:
            raise Exception("[plot_results.py] The data dimensions are wrong.");

        # Change the colour of the median lines
        medianprops = dict(color='black');

        # Plot the figure and add a grid
        fig = plt.figure(figsize=(10,5));
        ax = plt.axes();
        bplot1 = plt.boxplot(data,patch_artist=True,medianprops=medianprops,zorder=2);
        plt.grid(axis='y',**{"color":'white',"zorder":1});

        # Modifications to the graph
        plt.tick_params(axis=u'both',which=u'both',length=0,labelbottom=False);
        plt.yticks(**{"fontname":"Arial","fontsize":25});
        ax.set_facecolor("#EBECF3");

        for k in range(0,data.shape[1]):
            bplot1['boxes'][k].set_facecolor(colours_list[k]);

        # Set the x and y axes
        plt.xlabel(xlabel, **{"fontname":"Arial","fontsize":25});
        plt.ylabel(ylabel, **{"fontname":"Arial","fontsize":25});
        plt.title(title, **{"fontname":"Arial","fontsize":25});
        #plt.show();
        return fig;

    # Make a line plot
    def line_plot(self,data,xlabel,ylabel,title,colours_list):
        if type(data)!=np.ndarray or type(xlabel)!=str or type(ylabel)!=str or type(title)!=str:
            raise Exception("[plot_results.py] Wrong input");
        if data.shape[0]<=0 or data.shape[1]<=0 or len(colours_list)!=data.shape[1] or type(colours_list)!=list:
            raise Exception("[plot_results.py] The data dimensions are wrong.");

        # Plot the data
        fig = plt.figure(figsize=(12.5,7.5));
        ax = plt.axes();
        x = np.arange(2,len(data)+2,1,dtype=int);# The objects. We start with two objects in the scene
        plt.plot(x,data,'^',color='#8DB6D5');

        # Modifications to the graph
        plt.yticks(**{"fontname":"Arial","fontsize":25});
        plt.xticks(**{"fontname":"Arial","fontsize":25});
        ax.set_facecolor("#EBECF3");
        ax.set_xticklabels(x.astype(int))

        # Set the x and y axes
        plt.xlabel(xlabel, **{"fontname":"Arial","fontsize":25});
        plt.ylabel(ylabel, **{"fontname":"Arial","fontsize":25});
        plt.title(title, **{"fontname":"Arial","fontsize":25});
        return fig;

    # Run the plotting for the computational complexities experiments
    # @input folder String. The folder's absolute pathname containing all the results matrices
    # @input results_folder String. The folder's name containing all the results. It is not the absolute path, just the folder's name.
    # @input filenames_list List. A list of all the filenames containing the results.
    # @input resultsArray_list List containing of numpy arrays containing all the original data.
    def run_plotting_complexities(self,folder,results_folder,filenames_list,resultsArray_list):
        if type(folder)!=str or folder=="" or type(results_folder)!=str or results_folder=="" or type(filenames_list)!=list or len(filenames_list)<=0:
            raise Exception("[plot_results.py] Wrong input.");
        if len(resultsArray_list)<=0 or type(resultsArray_list)!=list or len(resultsArray_list)!=len(filenames_list):
            raise Exception("[plot_results.py] Wrong input.");
        
        # Append the word "_figures" to results_folder, so we can understand that the figures are saved in that folder
        results_folder = results_folder+"_figures";

        # Create a repository where all the plots are saved
        self.file_mangmt.create_folder(results_folder);

        # Take all the filenames inside the target folder and iterate over them
        row = 0;
        for file_path in filenames_list:
            data = np.array(resultsArray_list[row]).reshape(-1,1);
            figure_filepath = os.path.join(os.path.dirname(__file__),results_folder+'//');
            figure_filepath_line_conventional = figure_filepath + os.path.splitext(os.path.basename(file_path))[0]+"_conventional"+".png";
            fig = self.line_plot(data,"Number of Objects","Time (ms)","Task Planning Time",["#8DB6D5"]); # To complete
            self.save_fig(fig,figure_filepath_line_conventional);
            row = row + 1;
        return 0;
        
    # Plot and save all figures for all matrices containing the results
    # @input folder String. The folder's absolute pathname containing all the results matrices
    # @input results_folder String. The folder's name containing all the results. It is not the absolute path, just the folder's name.
    # @input filenames_list List. A list of all the filenames containing the results.
    # @input means_mat Numpy array. Array containing the execution times means to be plotted.
    # @input stderr_list List. List containing the original data's standard errors.
    # @input resultsArray_list List containing of numpy arrays containing all the original data.
    def run_plotting(self,folder,results_folder,filenames_list,means_mat,stderr_list,resultsArray_list):
        if means_mat.shape[0]<=0 or means_mat.shape[1]<=0 or means_mat.shape[0]!=len(stderr_list) or type(means_mat)!=np.ndarray or type(stderr_list)!=list or type(folder)!=str or folder=="" or type(results_folder)!=str or results_folder=="" or type(filenames_list)!=list or len(filenames_list)<=0:
            raise Exception("[plot_results.py] Wrong input.");
        if len(resultsArray_list)<=0 or type(resultsArray_list)!=list or len(resultsArray_list)!=len(filenames_list):
            raise Exception("[plot_results.py] Wrong input.");

        # The width of the bars. This is for plotting the bar plot
        width = 0.05; 

        # Append the word "_figures" to results_folder, so we can understand that the figures are saved in that folder
        results_folder = results_folder+"_figures";

        # Create a repository where all the plots are saved
        self.file_mangmt.create_folder(results_folder);
        
        # Take all the filenames inside the target folder and iterate over them
        row = 0;
        for file_path in filenames_list:
            means_row = means_mat[row,:].reshape(1,-1);
            stderr_row = stderr_list[row];
            figure_filepath = os.path.join(os.path.dirname(__file__),results_folder+'//');
            figure_filepath_bar_conventional = figure_filepath + os.path.splitext(os.path.basename(file_path))[0]+"_conventional"+".png";
            figure_filepath_bar_ours = figure_filepath + os.path.splitext(os.path.basename(file_path))[0]+"_ours"+".png";
            fig1 = self.bar_plot(means_row[:,[1,2,3]],stderr_row[:,[1,2,3]],["FD cg()","FD cg()+ff()","FD Astar cg()"],"","Average Time (ms)","Task Planning Time",["#8DB6D5","#C0A9CF","#98C5B2"],[0,width,2*width],width);
            fig2 = self.bar_plot(means_row[:,[0]],stderr_row[:,[0]],["Ours"],"","Average Time (ms)","Task Planning Time",["#CAA393"],[0],0.8);
            self.save_fig(fig1,figure_filepath_bar_conventional);
            self.save_fig(fig2,figure_filepath_bar_ours);

            tukey_box_plot_filepath_conventional = figure_filepath + os.path.splitext(os.path.basename(file_path))[0]+"_tukey_conventional"+".png";
            tukey_box_plot_filepath_ours = figure_filepath + os.path.splitext(os.path.basename(file_path))[0]+"_tukey_ours"+".png";
            original_data = resultsArray_list[row];
            fig1 = self.tukey_box_plot(original_data[:,[1,2,3]],"","Time (ms)","Task Planning Time",["#8DB6D5","#C0A9CF","#98C5B2"]);
            fig2 = self.tukey_box_plot(original_data[:,[0]],"","Time (ms)","Task Planning Time",["#CAA393"]);
            self.save_fig(fig1,tukey_box_plot_filepath_conventional);
            self.save_fig(fig2,tukey_box_plot_filepath_ours);
            row = row + 1;
        return 0;

def main():

    # Retrieve the filepath of the results file and get the 2D numpy array with all the numerical results
    # Instantiate a StatisticalAnalysis object
    results_folder = "results_comparison_14042023_Exp1_knownTasks";
    results_folder_computational_complexity = "results_computational_complexity";
    folder = os.path.join(os.path.dirname(__file__),results_folder+'//');
    folder_computational_complexity = os.path.join(os.path.dirname(__file__),results_folder_computational_complexity+'//');
    analysis = StatisticalAnalysis();
    graphics = Graphics();
   
    # Compute the statistical tests
    #results = analysis.run_analysis(folder);
    #post_hoc = results[5];
    #data = results[0];
    #means_mat = results[1];
    #stderr_list = results[3];
    #filenames_list = results[6];
    #resultsArray_list = results[7];

    # Get the results for plotting the computational complexities values
    #results = analysis.run_analysis_complexities(folder_computational_complexity);
    #resultsArray_list_compx = results[0];
    #filenames_list_compx = results[1];

    # Plot the results from the computational analysis
    #graphics.run_plotting_complexities(folder_computational_complexity,results_folder_computational_complexity,filenames_list_compx,resultsArray_list_compx);
    #exit(0);

    # Run through the results directory and save all the figures in another directory
    graphics.run_plotting(folder,results_folder,filenames_list,means_mat,stderr_list,resultsArray_list);

if __name__ == "__main__":
    main();