function opt = naoptimset(varargin)

opt = struct('MaxIterations', 20, ...
'InitialSample', 50, ...
'ReSamples', 40, ...
'CellsResampled', 10, ...
'MisfitBreak', NaN, ...
'NConstBreak', NaN, ...
'Verbose',false);


iarg = 1;
while iarg <= (length(varargin))
    switch lower(varargin{iarg})
        case 'maxiterations'
            opt.MaxIterations = varargin{iarg+1};
            iarg = iarg + 2;
            
        case 'initialsample'
            opt.InitialSample = varargin{iarg+1};
            iarg = iarg + 2;
            
        case 'resamples'
            opt.ReSamples = varargin{iarg+1};
            iarg = iarg + 2;
            
        case 'cellsresampled'
            opt.CellsResampled = varargin{iarg+1};
            iarg = iarg + 2;
            
        case 'misfitbreak'
            opt.MisfitBreak = varargin{iarg+1};
            iarg = iarg + 2;
            
        case 'nconstbreak'
            opt.NConstBreak = varargin{iarg+1};
            iarg = iarg + 2;
            
        case 'verbose'
            opt.Verbose = varargin{iarg+1};
            iarg = iarg + 2;
            
        otherwise
            error('NA:UnknownOption',...
                ['Unknown option: ',varargin{iarg}]);
    end
end

            
        
        
        
        