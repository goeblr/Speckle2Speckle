/**************************************************************************
SRAD_c speckle reducing anisotropic diffusion
    Y. Yu and S.T. Acton, "Speckle reducing anisotropic diffusion," IEEE
    Transactions on Image Processing, vol. 11, pp. 1260-1270, 2002.
    <http://viva.ee.virginia.edu/publications/j_srad.pdf>
 
MATLAB PROTOTYPE
    J = SRAD_c(single(I),niter,lambda,r1,r2,c1,c2,FLAG_SRAD_WAIT)

INPUTS
    I...............log compressed input image 
                    MATLAB = single, C = float
    niter...........number of iterations
    lambda..........smoothing time step
    r1/r2/c1/c2.....homogeneous ROI
                    upper left/bottom right row & col index, respectively
    FLAG_SRAD_WAIT..flag to use waitbar (1 = waitbar)
 
OUTPUT
    J...........log compressed output image
                MATLAB = single, C = float
 
USAGE
 
    J = SRAD_c(single(I),niter,lambda,r1,r2,c1,c2,FLAG_SRAD_WAIT)
    performs SRAD on the log compressed image I (of type "single" in 
    MATLAB, using NITER iterations and a timestep LAMBDA.  A homogeneous 
    region of interest (ROI) is defined via R1/R2/C1/C2, where R1/C1 is 
    the upper left ROI pixel and R2/C2 is the lower right ROI pixel.  
    The flag FLAG_SRAD_WAIT determines if we display a waitbar.

 WRITTEN BY:  Drew Gilliam & Rob Janiczek
 
 MODIFICATION HISTORY:
    2006.03   Rob Janiczek
        --creation of prototype version
    2006.03   Drew Gilliam
        --rewriting of prototype version into current version
        --got rid of multiple function calls, all code in a  
          single function (for speed)
        --code cleanup & commenting
        --code optimization efforts   
    2006.04   Drew Gilliam
        --added diffusion coefficent saturation on [0,1]

**************************************************************************/
#include <stdlib.h>
#include <math.h>
#include "mex.h"



void mexFunction(int nlhs, mxArray *plhs[],
int nrhs, const mxArray *prhs[] )
{
    /*=====================================================================
      VARIABLE DECLARATION
    =====================================================================*/
    
    // inputs
    float *I;                   // input image
    int niter;                  // nbr of iterations
    float lambda;               // update step size
    unsigned int r1,r2,c1,c2;   // row/col coordinates of uniform ROI
    bool FLAG_SRAD_WAIT;        // display waitbar
        
    // outputs
    float *J;               // output image ("single")
    
    // sizes
    unsigned int Nr,Nc,Ne;  // image nbr of rows/cols/elements
    unsigned int NeROI;     // ROI   nbr of elements
    
    // ROI statistics
    float meanROI, varROI, q0sqr ; //local region statistics
    
    // surrounding pixel indicies
    unsigned int *iN,*iS,*jE,*jW;    

    // center pixel value & directional derivatives
    float Jc, *dN,*dS,*dW,*dE;
    
    // calculation variables
    float tmp,sum,sum2;
    float G2,L,num,den,qsqr,D;
       
    // diffusion coefficient
    float *c, cN,cS,cW,cE;
    
    // counters
    int iter;   // primary loop
    int i,j;    // image row/col
    int k;      // image single index    
    
    // wait bar
    mxArray *waitset[2], *wait[2];
    double *xwait;
    

    /*=====================================================================
      SETUP
    =====================================================================*/
        
    // load input variables
    I      = (float*)mxGetPr(prhs[0]) ;
    niter  = (int)mxGetScalar(prhs[1]) ;
    lambda = (float)mxGetScalar(prhs[2]) ;
    r1     = (int)mxGetScalar(prhs[3]) - 1;
    r2     = (int)mxGetScalar(prhs[4]) - 1;
    c1     = (int)mxGetScalar(prhs[5]) - 1;
    c2     = (int)mxGetScalar(prhs[6]) - 1;
    FLAG_SRAD_WAIT = (bool)mxGetScalar(prhs[7]);
        
    // input image size
    Nr = mxGetM(prhs[0]);
    Nc = mxGetN(prhs[0]);
    Ne = Nr*Nc;
    
    // ROI image size    
    NeROI = (r2-r1+1)*(c2-c1+1);    
    
    // allocate local variables    
    iN = malloc(sizeof(unsigned int*)*Nr) ;
    iS = malloc(sizeof(unsigned int*)*Nr) ;
    jW = malloc(sizeof(unsigned int*)*Nc) ;
    jE = malloc(sizeof(unsigned int*)*Nc) ;    
    dN = malloc(sizeof(float)*Ne) ;
    dS = malloc(sizeof(float)*Ne) ;
    dW = malloc(sizeof(float)*Ne) ;
    dE = malloc(sizeof(float)*Ne) ;    
    c  = malloc(sizeof(float)*Ne) ;
        
    // N/S/W/E indices & boundary conditions
    for (i=0; i<Nr; i++) {
        iN[i] = i-1;
        iS[i] = i+1;
    }    
    for (j=0; j<Nc; j++) {
        jW[j] = j-1;
        jE[j] = j+1;
    }
    iN[0]    = 0;
    iS[Nr-1] = Nr-1;
    jW[0]    = 0;
    jE[Nc-1] = Nc-1;
           
    // allocate output variables
    plhs[0] = mxCreateNumericMatrix(Nr,Nc,mxSINGLE_CLASS,mxREAL);
    J = (float*)mxGetPr(plhs[0]);
        
    // waitbar setup
    if (FLAG_SRAD_WAIT) {
        
        // waitbar initial input - (x,'title')
        waitset[0] = mxCreateDoubleMatrix(1,1,mxREAL);
        waitset[1] = mxCreateString("SRAD: Diffusing Image");

        // waitbar input - (x,h)
        wait[0] = waitset[0];
        wait[1] = mxCreateDoubleMatrix(1,1,mxREAL);

        // waitbar current value
        xwait = mxGetPr(wait[0]);
        *xwait = 0;
        
        // open waitbar
        mexCallMATLAB(1,&wait[1],2,waitset,"waitbar");
    }
    
    
    /*=====================================================================
      SRAD
    =====================================================================*/
        
    // copy input to output & log uncompress
    for (k=0; k<Ne; k++) {
     	J[k] = (float)exp(I[k]) ;
    }
   
    // primary loop
    for (iter=0; iter<niter; iter++){
        
        // ROI statistics
        sum=0; sum2=0;
        for (i=r1; i<=r2; i++) {
            for (j=c1; j<=c2; j++) {
                tmp   = J[i + Nr*j];
                sum  += tmp ;
                sum2 += tmp*tmp;
            }
        }
        meanROI = sum / NeROI;
        varROI  = (sum2 / NeROI) - meanROI*meanROI;
        q0sqr   = varROI / (meanROI*meanROI);

        // directional derivatives, ICOV, diffusion coefficent
        for (j=0; j<Nc; j++) {
            for (i=0; i<Nr; i++) { 
                
                // current index/pixel
                k = i + Nr*j;
                Jc = J[k];
                
                // directional derivates
                dN[k] = J[iN[i] + Nr*j] - Jc;
                dS[k] = J[iS[i] + Nr*j] - Jc;
                dW[k] = J[i + Nr*jW[j]] - Jc;
                dE[k] = J[i + Nr*jE[j]] - Jc;
                     
                // normalized discrete gradient mag squared (equ 52,53)
                G2 = (dN[k]*dN[k] + dS[k]*dS[k] 
                    + dW[k]*dW[k] + dE[k]*dE[k]) / (Jc*Jc);

                // normalized discrete laplacian (equ 54)
                L = (dN[k] + dS[k] + dW[k] + dE[k]) / Jc;

                // ICOV (equ 31/35)
                num  = (0.5*G2) - ((1.0/16.0)*(L*L)) ;
                den  = 1 + (.25*L);
                qsqr = num/(den*den);
 
                // diffusion coefficent (equ 33)
                den = (qsqr-q0sqr) / (q0sqr * (1+q0sqr)) ;
                c[k] = 1.0 / (1.0+den) ;
                
                // saturate diffusion coefficent
                if (c[k] < 0) {c[k] = 0;}
                else if (c[k] > 1) {c[k] = 1;}
             
            }            
        }
        
        // divergence & image update
        for (j=0; j<Nc; j++) {
            for (i=0; i<Nr; i++) {        

                // current index
                k = i + Nr*j;
                
                // diffusion coefficent
                cN = c[k];
                cS = c[iS[i] + Nr*j];
                cW = c[k];
                cE = c[i + Nr*jE[j]];

                // divergence (equ 58)
                D = cN*dN[k] + cS*dS[k] + cW*dW[k] + cE*dE[k];
                
                // image update (equ 61)
                J[k] = J[k] + 0.25*lambda*D;
            }
        }
        
        // waitbar
        if (FLAG_SRAD_WAIT) {
            *xwait = (double)iter / (niter-1);
            mexCallMATLAB(0,NULL,2,wait,"waitbar");
        }
        
    } // end primary loop
    
    //log compress
    for (k=0; k<Ne; k++) {
     	J[k] = log(J[k]);
    }
        
    
    /*=====================================================================
      CLEANUP
    =====================================================================*/
    free(iN); free(iS); free(jW); free(jE);
    free(dN); free(dS); free(dW); free(dE);
    free(c);
    
    // waitbar cleanup
    if (FLAG_SRAD_WAIT) {
        mexCallMATLAB(0,NULL,1,&wait[1],"close");
        mxDestroyArray(waitset[0]);
        mxDestroyArray(waitset[1]);
        mxDestroyArray(wait[1]);
    }      
    
    return;
    
} // end mexFunction()



/**************************************************************************
  END OF FILE
**************************************************************************/
