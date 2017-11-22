# PySFC
Python implementation of the SFC intonation model.
 

# The SFC model

The Superposition of Functional Contours (SFC) model is a prosody model that is based on the decomposition of prosodic contours into functionally relevant elementary contours [1]. It proposes a generative mechanism for encoding socio-communicative functions, such as syntactic structure and attitudes, through the use of prosody. 
The SFC has been successfully used to model different linguistic levels, including: attitudes, dependency relations of word groups (dependency to the left/right of a verb group to its subject/object noun group), cliticisation, narrow word focus, tones in Mandarin, and finally syntactic relations between operands and operators in math. It has been also successfully applied to a number of languages including: French, Galician, German and Chinese. Recently, the SFC model has been extended into the visual prosody domain, where it has been used to analyse and generate facial expressions and head and gaze motion control, to communicate speaker attitude. 

The SFC model is based on neural network contour generators (NNCGs) that are trained to learn elementary prosodic contours. Each NNCG is responsible for encoding one linguistic function on one given scope. The end multiparametric prosody contour is then obtained by overlapping and adding these elementary contours according to their scopes within the utterance. 
The NNCG training algorithm comprises of an analysis-by-synthesis loop that distributes the error at every iteration among the contour generators that are active for each of the syllables. Their previous output is then adjusted by the error and used as a target for the next iteration of backpropagation training.

The NNCGs used in the SFC comprise a shallow feedforward neural network with one hidden layer, with a default number of 15 neurons. Four syllable position ramps are used as input that capture absolute and relative position of the syllable within the global scope of the linguistic function and also locally, i.e. in the left and right scopes as divided by the central landmark in some functions. The NNCG uses these input ramps to generate pitch and duration coefficients for each of the syllables.


# PySFC

PySFC is a Python implementation of the SFC model that was created with two goals: i) to make the SFC more accessible to the scientific community, and ii) to serve as a foundation for future improvements of the prosody model. 
The PySFC also implements a minimum set of tools necessary to make the system self-contained and fully functional. These are, for the most part, used for working with the data and plotting. 

Python was chosen as an implementation language because of the powerful scientific computing environment that is completely based on free software and that has recently gained power and momentum in the scientific community. The environment is based on the fundamental [NumPy](http://www.numpy.org/) scientific computing package within the [SciPy](https://www.scipy.org/) scientific Python ecosystem.

To enhance the accessibility of the PySFC to the broader community, great attention was put on code readability, which is also one of the prime features of good Python, augmented with detailed functions docstrings, and the generous use of comments. The code is segmented in [Spyder](https://pythonhosted.org/spyder/) cells for rapid prototyping. Finally, the whole implementation has been licensed as [free software](http://fsf.org/) with a [GNU General Public License v3](http://www.gnu.org/licenses/).
and has been made available on GitHubhttps://github.com/bgerazov/PySFC.

# PySFC Modules

PySFC comprises the following modules:
 * `sfc.py` - main module that controls the application of the SFC model to a chosen dataset. It is responsible for using all of the other modules to read the data, train the SFC model on it and output plots and statistics. To facilitate debugging it features elaborate logging of code execution. 
 * `sfc_params.py` - parameter setting module that holds all the users settable parameters including:
      * data related parameters - type of input data, phrase level functional contours (attitudes), local functional contours (e.g. emphasis, syntactic dependencies), 
      * SFC hyperparameters - number of points to be sampled from the pitch contour at each vowel nucleus, pitch and duration coefficient scaling coefficients, number of iterations for analysis-by-synthesis, as well as the number of hidden units, learning rate, maximum number of iterations and regularisation coefficient for the NNCGs, 
      * SFC execution parameters - use of preprocessed data, use of trained models, plotting functionality.
 * `sfc_corpus.py` - holds all the functions that are used to consolidate and work with the corpus of data that is directly fed and output from the SFC model. This data corpus includes the features extracted from the input data and the output predictions of the SFC. The corpus is stored in a [Pandas](http://pandas.pydata.org/) data frame object, which allows easy data access and analysis. The largest part of this module is the function build_corpus() which reads the input files and constructs the corpus.
 * `sfc_data.py` - comprises functions that read the input data files and calculate the `f_0` and duration coefficients input to the SFC. The module also holds functions that generate intonation, and phone and syllable duration statistics. 
 * `sfc_learn.py` - holds the SFC training function `analysis_by_synthesis()` and the function for NNCG initialisation. The neural networks and their training have been facilitated through the use of the Multi Layer Perceptron (MLP) regressor in the powerful [scikit-learn](http://scikit-learn.org/stable/index.html) module. 
 * `sfc_dsp.py` - holds DSP functions for smoothing the pitch contour based on SciPy.
 * `sfc_plot.py` - holds the plotting functions based on [matplotlib](http://matplotlib.org/) and [seaborn](http://seaborn.pydata.org/) that are used to plot the data analysis, synthesised SFC contours and SFC losses.

Currently, PySFC supports the proprietary SFC `fpro` file format as well as standard Praat `TextGrid` annotations. Pitch is calculated based on Praat `PointProcess` pitch mark files, but integration of state-of-the-art pitch extractors is planned for the future. 

PySFC also brings added value, by adding the possibility to adjust the number of samples to be taken from the pitch contour at each rhythmical unit vowel nucleus, and with its extended plotting capabilities for data and performance analysis.

# PySFC example plots