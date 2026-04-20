%
% function [model_opt,mfitmin,na_models2D,workNA2] = ...
%   NeighbourhoodAlgorithm(ObjFunc,nd,lb,ub,varargin)
%
% Nearest Neighbourhood Minimisation Function
% 
% The NN algorithm uses Voronoi cells to search a parameter space to
% minimise a function. Method is described by:
%
% Sambridge M. 1999, Geophysical inversion with a neighbourhood algorithm -
%   I. Searching a parameter space: G.J.I. 138, 479-494.
%
% This code is a Matlab-isation of Sambridge's original Fortran code. 
% Therefore in some ways it retains a Fortran 'flavour' to it, and could
% probably stand to be improved into proper Matlab-ese. 
%
% This version is based on the Bristol University version of the NN code, 
% correcting a number of bugs, improving user-controlled I/O, stripping
% many "dead-wood" options that are rarely used, and also much other 
% "dead-wood" that doesn't seem to do anything useful in the code
%
% INPUTS:   ObjFunc = handle to the objective function to be minimised
%           nd = number of parameter space dimensions to be searched
%           lb = lower bounds of parameter space
%           ub = upper bounds of parameter space
%
% OPTIONAL INPUTS
%           optional inputs control how the search proceeds. See Sambridge
%               (1999) for more details
%           MaxIterations = maximum number of iterations in the search
%               (default = 20)
%           InitialSample = number of initial sample points
%               (default = 50)
%           ReSamples = number of resamples in each new iteration
%               (default = 40)
%           CellsResampled = number of cells to be resampled in the next 
%               iteration
%               (default = 10)
%
% EXTRA OPTIONAL INPUTS
%           extra inputs additional to the Sambridge code
%           MisfitBreak = value of ObjFunc, once reached the inversion 
%               stops 
%           NConstBreak = number of iterations after which, if the minimum
%               misfit hasn't improved, the inversion stops
%
% These input parameters can be defined individually using keywords in the
% command line, or via a structure created by the naoptimset command. To 
% use a structure for options, this should be the sole option set. 
%
% OUTPUTS:  model_opt = best fitting model
%           mfitmin = ObjFunc value at the optimum model position
%           na_modelsND = list of all models tested
%           workNA2 = ObjFunc values at each position of na_modelsND
%
% Written by J.P. Verdon, UoB, based on Fortran code by M. Sambridge, ANU,
% which has been updated by J. Wookey, UoB and J.P. Verdon
%

function [model_opt,mfitmin,na_modelsND,workNA2] = ...
    NA_point(ObjFunc,nd,lb,ub,itmax,nsamplei,nsample,ncells,mfbreak)

%tic  % Start timing

% Process the optional arguments (overwriting defaults where defined)
ncbreak = NaN;
verbose = false;

% Maximum size of arrays
nsample_max = 100;
nit_max = 5000;
nmod_max = nsample_max*(nit_max+1);

% Hardwired input parameters
nsleep = 1;
nclean = 500;
iproc = 0;
nproc = 1;

% Initialise some matrices
misfit = NaN(nmod_max,1);
workNA1 = NaN(nmod_max,1);
workNA2 = NaN(nmod_max,1);
mopt = 0;
mfitmin = 0;
mfprev = 0;
mfbreakcount = 0;

% Defunct parameters from Fortran code
%nd_max = 10;
%nh_max = 1000;
%nsleep_max = 1;
%maxseq = 50;
%infolevel = 0;
%nsam = max(nsample,nsamplei);
%ntotal = nsamplei + nsample*itmax;
%istype = 0;
%iworkNA1 = NaN(nmod_max,1);
%iworkNA1 = NaN(nsample_max,1);


% Initialise upper and lower bounds and ranges
ranges(1,:) = lb;
ranges(2,:) = ub;
scales(1:nd) = -1;


%call NA_initialize(range,ranget,scales,nd,xcur,nsample,ncells,restartNA)
[ranget,scales,xcur,ic,restartNA] = NA_Initialize(ranges,scales,nd);

% Generate initial samples
%call NA_initial_sample(na_models,nd,ranget,range,nsamplei,numberOfModels,scales,misfit)
na_models = NA_InitialSample(nd,nsamplei,ranget);


ntot = 0;
ns = nsamplei;


for it = 1:itmax + 1
    if it > 1
        if verbose
            fprintf('%s \n',['Iteration: ',num2str(it),' NTot: ',num2str(ntot),...
                ' MinMF: ',num2str(mfitmin)]);
        end
            
    end
    % Calculate misfit values for each model in the current population
    for i = iproc+1:nproc:ns
        
        % Decode current model and put into array model
        ii = 1+(i-1+ntot)*nd;
        
        na_model = transform2raw(na_models(ii:ii+(nd-1)),nd,ranges,scales);
        
        % Generate the forward model and compute misfit        
        misfitval = ObjFunc(na_model);
        
        jj = ntot + i;       
        
        misfit(jj) = misfitval;
        
    end
    
    % Calculate properties of current misfit distribution (mean, min, best
    % model, etc)
    [mopt,mfitmin,mfitord,workNA2] = ...
    NA_Misfits(misfit,ns,it,ntot,mfitmin,mopt,ncells);
    
    ii = 1 + (mopt-1)*nd;
    model_opt = transform2raw(na_models(ii:ii+(nd-1)),nd,ranges,scales) ;
    
    ntot = ntot + ns;
    ns = nsample;
    
    if it == itmax + 1
        break
    end
    % Escape if MF is below mfbreak
    if isfinite(mfbreak) && mfbreak > mfitmin
        if verbose
            fprintf('%s \n',['Minimum misfit ',num2str(mfitmin),' has gone below ',...
                'MisfitBreak cutoff of ',num2str(mfbreak),'. Neighbourhood Algorithm ',...
                'is stopping'])
        end
        break
    end
    % Check to see whether misfit is improving
    if isfinite(ncbreak)
        if mfprev == mfitmin
            mfbreakcount = mfbreakcount + 1;
        else
            mfbreakcount = 0;
        end
        
        if mfbreakcount >= ncbreak
            if verbose
                fprintf('%s \n',['Misfit of ',num2str(mfitmin),' has not improved for ',...
                    num2str(ncbreak),' iterations. Neighbourhood Algorithm ',...
                    'is stopping'])
            end
            break
        end
        mfprev = mfitmin; 
    end
        
        
        
    % Generate a new sample using Neighbourhood algorithm (resample version)     
    [na_models,xcur,ic] = NA_Sample(na_models,ntot,nsample,nd,nsleep,ncells,...
        mfitord,ranget,xcur,restartNA,nclean,workNA1,ic);
           
  
end

% transform all models back to scaled units
for i=1:ntot
	ii = 1 + (i-1)*nd;
    tmod(ii:ii+(nd-1)) = transform2raw(na_models(ii:ii+(nd-1)),nd,ranges,scales);
    
    
end

na_modelsND = NaN(nd,ntot);
count = 0;
for i = 1:ntot
    for j = 1:nd
        count = count+1;
        na_modelsND(j,i) = tmod(count);
    end
end
    

%toc
        