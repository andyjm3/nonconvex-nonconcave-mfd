function [xy, mycost, info, options] = rsm(problem, xy, options)
% Riemannian second-order methods for min-max optimization

% function [xy, cost, info, options] = rsm(problem)
% function [xy, cost, info, options] = rsm(problem, xy)
% function [xy, cost, info, options] = rsm(problem, xy, options)
% function [xy, cost, info, options] = rsm(problem, [], options)

    M = problem.M;

    % Verify that the problem description is sufficient for the solver.
    if ~canGetCost(problem)
        warning('manopt:getCost', ...
                'No cost provided. The algorithm will likely abort.');  
    end
    if ~canGetGradient(problem) && ~canGetApproxGradient(problem)
        % Note: we do not give a warning if an approximate gradient is
        % explicitly given in the problem description, as in that case the user
        % seems to be aware of the issue.
        warning('manopt:getGradient:approx', ...
               ['No gradient provided. Using an FD approximation instead (slow).\n' ...
                'It may be necessary to increase options.tolgradnorm.\n' ...
                'To disable this warning: warning(''off'', ''manopt:getGradient:approx'')']);
        problem.approxgrad = approxgradientFD(problem);
    end
    if ~canGetHessian(problem) && ~canGetApproxHessian(problem)
        % Note: we do not give a warning if an approximate Hessian is
        % explicitly given in the problem description, as in that case the user
        % seems to be aware of the issue.
        warning('manopt:getHessian:approx', ...
               ['No Hessian provided. Using an FD approximation instead.\n' ...
                'To disable this warning: warning(''off'', ''manopt:getHessian:approx'')']);
        problem.approxhess = approxhessianFD(problem);
    end
    
    
    localdefaults.verbosity = 2;
    localdefaults.maxtime = inf;
    localdefaults.maxiter = 100;
    localdefaults.tolgradnorm = 1e-10; % tolgradnorm on f
    localdefaults.stepsize_x = 0.2;
    localdefaults.stepsize_y = 0.2;
    localdefaults.inv_method = 'cg'; % either cg or ls 
    localdefaults.epscg = 0; % eps regularization for solving inverse
    localdefaults.zeta = 1; % stepsize multiplier for the newton step (nfr, ntgda)
    localdefaults.inv_tol = 1e-6;
    localdefaults.inv_iter = 20;
    localdefaults.mode = 'fr'; % fr, nfr, tgda, ntgda, ngd
    localdefaults.update = 'retr';
    
    % Merge global and local defaults, then merge w/ user options, if any.
    localdefaults = mergeOptions(getGlobalDefaults(), localdefaults);
    if ~exist('options', 'var') || isempty(options)
        options = struct();
    end
    options = mergeOptions(localdefaults, options);
    
    
    stepsize_x = options.stepsize_x;
    stepsize_y = options.stepsize_y;
    stepsize = stepsize_y;
    tau = stepsize_x/stepsize_y;
    epscg = options.epscg;
    algname = options.mode;
    inv_method = options.inv_method;
    inv_tol = options.inv_tol;
    inv_iter = options.inv_iter;
    
    zeta = options.zeta; % nfr, ntgda
    
    
    
    start_time = tic();
    
    % If no initial point x is given by the user, generate one at random.
    if ~exist('xy', 'var') || isempty(xy)
        xy = M.rand();
    end
        
    % Create a store database and get a key for the current x
    storedb = StoreDB(options.storedepth);
    key = storedb.getNewKey();
    
    % Initialize solution
    [mycost, gradfxy] = getCostGrad(problem, xy, storedb, key);
    gradnorm = M.norm(xy, gradfxy);
    
    
    % savestats
    iter = 0;
    savedstats = 0;
    stats = savestats();
    info(1) = stats;
    savedstats = savedstats + 1;
    preallocate = options.maxiter + 1;
    info(preallocate).iter = [];
    
    % Display information header for the user.
    if options.verbosity >= 1
        fprintf('\n-------------------------------------------------------\n');
        fprintf('r%s:  iter\t               gradnorm \t     stepsize\n', algname);
        fprintf('r%s:  %5d\t%+.16e\t%.8e\n', algname, iter, gradnorm, stepsize);
    end
    
    
    elapsed_time = 0;
    for iter = 1:options.maxiter
        % Record start time.
        start_time = tic();
        
        if strcmp(algname, 'fr')
            % compute direction x
            gradfxy_mod.x = problem.Mx.lincomb(xy.x,-tau,gradfxy.x);

            % compute direction y
            gradyx_gradx = problem.hessyx(xy, gradfxy.x);
            gradfmod_y = hessinv(xy, gradyx_gradx, inv_method);
            gradfxy_mod.y = problem.My.lincomb(xy.y,1,gradfxy.y,tau,gradfmod_y);

            update_v = gradfxy_mod;

            % gradient update
            if strcmp(options.update, 'exp') && isfield(problem.M, 'exp')
                newxy =  problem.M.exp(xy, update_v, stepsize);
            elseif strcmp(options.update, 'retr')
                newxy =  problem.M.retr(xy, update_v, stepsize);
            end
            
        elseif strcmp(algname, 'tgda')
            % compute direction x
            gradx_mod = problem.Mx.lincomb(xy.x,-tau,gradfxy.x);
            hessgrad_mod = hessinv(xy, gradfxy.y, inv_method);
            hessgrad_mod = problem.hessxy(xy, hessgrad_mod);
            gradfxy_mod.x = problem.Mx.lincomb(xy.x, 1, gradx_mod, tau, hessgrad_mod);

            % compute direction y
            gradfxy_mod.y = gradfxy.y;

            update_v = gradfxy_mod;

            % gradient update
            if strcmp(options.update, 'exp') && isfield(problem.M, 'exp')
                newxy =  problem.M.exp(xy, update_v, stepsize);
            elseif strcmp(options.update, 'retr')
                newxy =  problem.M.retr(xy, update_v, stepsize);
            end
            
        elseif strcmp(algname, 'ngd')
            % compute direction x and update x
            gradfx_mod = problem.Mx.lincomb(xy.x,-tau,gradfxy.x);
            if strcmp(options.update, 'exp') && isfield(problem.Mx, 'exp')
                newx =  problem.Mx.exp(xy.x, gradfx_mod, stepsize);
            elseif strcmp(options.update, 'retr')
                newx =  problem.Mx.retr(xy.x, gradfx_mod, stepsize);
            end
            xy.x = newx;

            % compute direction y
            gradfxynew = getGradient(problem, xy);
            hessgrad_mod = hessinv(xy, gradfxynew.y, inv_method);
            hessgrad_mod = problem.My.lincomb(xy.y,-1,hessgrad_mod);
            if strcmp(options.update, 'exp') && isfield(problem.Mx, 'exp')
                newy =  problem.My.exp(xy.y, hessgrad_mod, stepsize);
            elseif strcmp(options.update, 'retr')
                newy =  problem.My.retr(xy.y, hessgrad_mod, stepsize);
            end
            xy.y = newy;
            newxy = xy;
            
        elseif strcmp(algname, 'nfr')
            % compute direction x
            gradfxy_mod.x = problem.Mx.lincomb(xy.x,-tau,gradfxy.x);

            % compute direction y
            gradyx_gradx = problem.hessyx(xy, gradfxy.x);
            rhs = problem.My.lincomb(xy.y, -zeta, gradfxy.y, tau, gradyx_gradx);
            gradfxy_mod.y = hessinv(xy, rhs, inv_method);

            update_v = gradfxy_mod;

            % gradient update
            if strcmp(options.update, 'exp') && isfield(problem.M, 'exp')
                newxy =  problem.M.exp(xy, update_v, stepsize);
            elseif strcmp(options.update, 'retr')
                newxy =  problem.M.retr(xy, update_v, stepsize);
            end
            
        elseif strcmp(algname, 'ntgda')
            % compute direction x
            gradx_mod = problem.Mx.lincomb(xy.x,-tau,gradfxy.x);
            hessgrad_mod = hessinv(xy, gradfxy.y, inv_method);
            hessgrad_mod = problem.hessxy(xy, hessgrad_mod);
            gradfxy_mod.x = problem.Mx.lincomb(xy.x, 1, gradx_mod, tau, hessgrad_mod);

            % compute direction y
            grady_mod = problem.My.lincomb(xy.y,-zeta,gradfxy.y);
            gradfxy_mod.y = hessinv(xy, grady_mod, inv_method);

            update_v = gradfxy_mod;

            % gradient update
            if strcmp(options.update, 'exp') && isfield(problem.M, 'exp')
                newxy =  problem.M.exp(xy, update_v, stepsize);
            elseif strcmp(options.update, 'retr')
                newxy =  problem.M.retr(xy, update_v, stepsize);
            end
            
        else
            error('Invalid algorithm mode!');
        end
        
        
        % Make new key
        newkey = storedb.getNewKey();
        xy = newxy;
        key = newkey;

        %compute grad, gradnorm
        [mycost, gradfxy] = getCostGrad(problem, xy, storedb, key);
        gradnorm = problem.M.norm(xy, gradfxy);

        % == stats save ==
        % Log statistics for freshly executed iteration.
        elapsed_time = elapsed_time + toc(start_time);
        stats = savestats();
        info(savedstats+1) = stats;
        savedstats = savedstats + 1;
        
        % Reset timer.
        elapsed_time = 0;

        if options.verbosity >= 1
            fprintf('%s:  %5d\t%+.16e\t%.8e\n', algname, iter, gradnorm, stepsize);
        end
        
        if gradnorm < options.tolgradnorm
            break;
        end
        
        
    end
    
    % Keep only the relevant portion of the info struct-array.
    info = info(1:savedstats); 
    
    
    % Helper function to collect statistics 
    function stats = savestats()
        stats.iter = iter;
        if savedstats == 0
            stats.time = toc(start_time);
            stats.stepsize = NaN;
            stats.cost = mycost;
            stats.gradnorm = gradnorm;
        else
            stats.time = info(savedstats).time + elapsed_time;
            stats.stepsize = stepsize;
            stats.cost = mycost;
            stats.gradnorm = gradnorm;
        end
        stats = applyStatsfun(problem, xy, storedb, key, options, stats);
    end
    
    % This should be moved outside of function?
    % compute hessy inv
    function hinv = hessinv(xy, rhs, inv_method)
        if strcmp(inv_method, 'cg')
            % can lead to ill-conditioned due to the projection
            rhs = problem.hessyy(xy, rhs); %%
            hinv = pcg(@(v)hesssqv(v, []), rhs(:), inv_tol, inv_iter);
            hinv = problem.My.mat(xy.y, hinv);
            hinv = problem.My.proj(xy.y, hinv);            
            
        elseif strcmp(inv_method, 'tscg')
            hinv = ts_cg(problem.My,xy.y,@(v)problem.hessyy(xy, v),rhs,epscg,inv_tol, inv_iter);
            
            %disp(problem.My.norm(xy.y, problem.hessyy(xy, hinv) - rhs));
        elseif strcmp(inv_method, 'ls')
            rhs = problem.hessyy(xy, rhs); %%
            hinv = lsqr(@hesssqv, rhs(:), inv_tol, inv_iter);
            hinv = problem.My.mat(xy.y, hinv);
            hinv = problem.My.proj(xy.y, hinv);
        end
    end
    

    
    function hv = hesssqv(v, opt)
        % hess_y squared applied on v
        V = problem.My.mat(xy.y, v);
        V = problem.My.proj(xy.y, V);
        hv = problem.hessyy(xy, V);
        hv = problem.hessyy(xy, hv);
        hv = problem.My.lincomb(xy.y, (1+epscg), hv);
        hv = hv(:);
    end

    
end

