# A Microplastics Treatment Evaluation Routine 

In this repository we analyse pairs of microscopic images containing bright particles on dark background.
The code was created for a study developing an evaluation concept for microplastic purification protocols.

**The original publication can be found here: [...soon]**

## The concept in a nutshell
Microplastic particles are immobilised on a plane silicon wafer substrate using a micrometer thin spin coated layer of epoxy resin.
After curing, the wafer is imaged, exposed to a treatment that needs evaluation and subsequently imaged again.
The pre- and post-treatment images are compared with respect to changes in particle areas and numbers caused by the treatment.


## Usage guideline
There are two main parts in the analysis of the image:

- identifying, matching and measuring particles
- evaluating and plotting the results

The first part is started by running the `compareDigestImages.py` where appropriate paths to the pre-post-image directories need to be entered
To replicate the original study's analysis, all required image data set can be obtained from: [...DOI]
The results are saved in a set of csv files which contain the data of the particle measurements.

In the second part these csv files are read by running the `analyse_data.py` in the `data_analysis` subdirectory.
Finally, the results are visualised for further exploration in an interactive figure (based on the [Altair python library](https://github.com/altair-viz/altair)) and saved as an html file.

The output figure used in the original publication can be found at: [...DOI].
