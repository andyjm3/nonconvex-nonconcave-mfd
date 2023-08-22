function test_spd_quadratic_dist()
    % test on nonconvex-nonconcave quadratic optimization
    % on SPD manifold with Riemannian distance on Y
    % f(X,Y) = c1 * (logdet(X))^2 + c2 * logdet(X)logdet(Y) + c3 *
    % (dist(I, Y))^2
    
    clear
    clc
    rng(42);
    
    d = 30;
    imat = eye(d);
    symm = @(X) .5*(X+X');

    setup = 1; % nonconvex-nonconcave
    
    switch setup
        case 1 
            c1 = -1;
            c2 = 5;
            c3 = -1;
            
    end

    %% problem define
    problem.Mx = sympositivedefinitefactory(d);
    problem.My = sympositivedefinitefactory(d);
    problem.M = productmanifold(struct('x', problem.Mx, 'y', problem.My));
    
    problem.cost = @cost;
    problem.grad = @grad;
    problem.hessxx = @hessxx; % compute the hessian 4 components
    problem.hessyy = @hessyy;
    problem.hessxy = @hessxy;
    problem.hessyx = @hessyx;
    problem.hess = @hess;

    function f = cost(xy)
        x = xy.x;
        y = xy.y;
        logdetx = log(det(x));
        logdety = log(det(y));
        f = c1 *logdetx^2 + c2 * logdetx*logdety + c3* norm(logm(xy.y), 'fro')^2;
        f = f/d;
    end

    function g = grad(xy)
        x = xy.x;
        y = xy.y;
        logdetx = log(det(x));
        logdety = log(det(y));
        g.x = (2 * c1 * logdetx + c2 *logdety) .* x;
        g.x = g.x/d;
        %g.y = (c2 * logdetx) .* y - 2 * c3 * problem.My.log(xy.y,imat);
        g.y = (c2 * logdetx) .* y - 2 * c3 * symm(y * logm(y \ imat));
        g.y = g.y/d;
    end 

    function hxx = hessxx(xy, xdot)
        x = xy.x;
        hxx = 2 * c1 * trace(x \ xdot) .* x;
        hxx = hxx/d;
    end

    function hyy = hessyy(xy, ydot)
        y = xy.y;
        grady = - 2 * c3 * symm(y * logm(y \ imat));
        grady = grady/d;
        hyy = (- 2 * c3/d) * symm(ydot * logm(y \ imat) - y * dlogm(y \imat, y \ ydot / y)); % Dgrady
        hyy = hyy - symm(ydot * (y \ grady)); % correction term
    end

    function hxy = hessxy(xy, ydot)
        x = xy.x;
        y = xy.y;
        hxy = c2 * trace(y \ ydot) .* x; 
        hxy = hxy/d;
    end

    function hyx = hessyx(xy, xdot)
        x = xy.x;
        y = xy.y;
        hyx = c2 * trace(x \ xdot) .* y;
        hyx = hyx/d;
    end

    function hcp = hesscp(xy, xydot)
        x = xy.x;
        y = xy.y;
        xdot = xydot.x;
        ydot = xydot.y; 
        
        hcp.xx = hessxx(xy, xdot);
        hcp.xy = hessxy(xy, ydot);
        hcp.yx = hessyx(xy, xdot);
        hcp.yy = hessyy(xy, ydot);
        
    end

    function h = hess(xy, xydot)
        hcp = hesscp(xy, xydot);
        
        h.x = hcp.xx + hcp.xy;
        h.y = hcp.yy + hcp.yx;        
    end
    %checkgradient(problem);
    %checkhessian(problem);
    %keyboard;
    

    xy0 = problem.M.rand();
    maxiter = 200;
    
    
    function stats = computestats(problem, xy, stats)
        stats.disttoopt = abs(det(xy.x) - 1) + norm(logm(xy.y), 'fro')^2;
    end
    
    % compute stats function for RHM
    function stats = computestatsRHM(problem, xy, stats)
        stats.gradnormf = sqrt(2*stats.cost);
        stats.disttoopt = abs(det(xy.x) - 1) +  norm(logm(xy.y), 'fro')^2;
    end


    %%
    
    % RGDA
    optionsRGDA = set_options('RGDA');
    [~,~,info_rgda,~] = rgda(problem, xy0, optionsRGDA);
    

    %
    % RTSGDA
    optionsRGDA = set_options('TSRGDA');
    [~,~,info_tsrgda,~] = rgda(problem, xy0, optionsRGDA);
    
    
    
    % RHM
    problemHGD.Mx = sympositivedefinitefactory(d);
    problemHGD.My = sympositivedefinitefactory(d);
    problemHGD.M = productmanifold(struct('x', problemHGD.Mx, 'y', problemHGD.My));
    problemHGD.cost = @cost;
    problemHGD.grad = @grad;
    problemHGD.hess = @hess;
    optionsRHMsdf = set_options('RHM');
    [~,~,info_rhg_sdf,~] = rhm(problemHGD, xy0, optionsRHMsdf);
    
    
    % RCON
    optionsRHMcon = set_options('RCON');
    [~,~,info_rhg_con,~] = rhm(problemHGD, xy0, optionsRHMcon);  
    %}
        
    % RCEG
    optionsRCEG = set_options('RCEG');
    [~,~,info_rceg,~] = rceg(problem, xy0, optionsRCEG);
    
    % RFR
    optionsRFR = set_options('RFR');
    [~ ,~,infor_rfr,~] = rsm(problem, xy0, optionsRFR);
    
    
    % RTGDA
    optionsRTGDA = set_options('RTGDA');
    [~,~,infor_rtgda,~] = rsm(problem, xy0, optionsRTGDA);
    %}

    % RNGD
    optionsRNGD = set_options('RNGD');
    [~,~,infor_rngd,~] = rsm(problem, xy0, optionsRNGD); 
    %}

    % RNFR
    optionsRNFR = set_options('RNFR');
    [~,~,infor_rnfr,~] = rsm(problem, xy0, optionsRNFR);
    
    % RNTGDA
    optionsRNTGDA = set_options('RNTGDA');
    [~,~,infor_rntgda,~] = rsm(problem, xy0, optionsRNTGDA);

    %%
    colors = [[247, 129, 191]/255; [166, 86, 40]/255; [255, 127, 0]/255];
    colors = cat(1,colororder, colors);
    lw = 1.3;
    ms = 3.5;

    % iter
    h1 = figure(1);
    semilogy([info_rgda.iter], [info_rgda.disttoopt], '->', 'MarkerSize',ms, 'LineWidth',lw, 'color', colors(1,:)); hold on;
    semilogy([info_tsrgda.iter], [info_tsrgda.disttoopt], '-*', 'MarkerSize',ms, 'LineWidth',lw, 'color', colors(2,:)); hold on;
    semilogy([info_rceg.iter], [info_rceg.disttoopt], '-o', 'MarkerSize',ms, 'LineWidth',lw, 'color', colors(3,:)); hold on;
    semilogy([info_rhg_sdf.iter], [info_rhg_sdf.disttoopt], '-d', 'MarkerSize',ms, 'LineWidth',lw, 'color', colors(4,:)); hold on;
    semilogy([info_rhg_con.iter], [info_rhg_con.disttoopt], '-+', 'MarkerSize',ms, 'LineWidth',lw, 'color', colors(5,:)); hold on;
    semilogy([infor_rfr.iter], [infor_rfr.disttoopt], '-s', 'MarkerSize',ms, 'LineWidth',lw, 'color', colors(6,:)); hold on;
    semilogy([infor_rtgda.iter], [infor_rtgda.disttoopt], '-^', 'MarkerSize',ms, 'LineWidth',lw, 'color', colors(7,:)); hold on;
    semilogy([infor_rngd.iter], [infor_rngd.disttoopt], '-x', 'MarkerSize',ms, 'LineWidth',lw, 'color', colors(8,:)); hold on;
    semilogy([infor_rnfr.iter], [infor_rnfr.disttoopt], '-diamond', 'MarkerSize',ms, 'LineWidth',lw, 'color', colors(9,:)); hold on;
    semilogy([infor_rntgda.iter], [infor_rntgda.disttoopt], '-square', 'MarkerSize',ms, 'LineWidth',lw, 'color', colors(10,:)); hold on;
    hold off;
    ax = gca;
    lg = legend({'RGDA', 'TSRGDA', 'RCEG', 'RHM', 'RCON', 'RFR', 'RTGDA', 'RNGD', 'RNFR', 'RNTGDA'}, 'NumColumns',2);
    lg.FontSize = 14;
    xlabel(ax,'Iteration','FontSize',22);
    ylabel(ax,'Optimality gap','FontSize',22);
    
    
    % time
    h2 = figure(2);
    semilogy([info_rgda.time], [info_rgda.disttoopt], '->', 'MarkerSize',ms, 'LineWidth',lw, 'color', colors(1,:)); hold on;
    semilogy([info_tsrgda.time], [info_tsrgda.disttoopt], '-*','MarkerSize',ms, 'LineWidth',lw, 'color', colors(2,:)); hold on;
    semilogy([info_rceg.time], [info_rceg.disttoopt], '-o','MarkerSize',ms, 'LineWidth',lw, 'color', colors(3,:)); hold on;
    semilogy([info_rhg_sdf.time], [info_rhg_sdf.disttoopt], '-d','MarkerSize',ms, 'LineWidth',lw, 'color', colors(4,:)); hold on;
    semilogy([info_rhg_con.time], [info_rhg_con.disttoopt], '-+','MarkerSize',ms, 'LineWidth',lw, 'color', colors(5,:)); hold on;
    semilogy([infor_rfr.time], [infor_rfr.disttoopt], '-s','MarkerSize',ms, 'LineWidth',lw, 'color', colors(6,:)); hold on;
    semilogy([infor_rtgda.time], [infor_rtgda.disttoopt], '-^','MarkerSize',ms, 'LineWidth',lw, 'color', colors(7,:)); hold on;
    semilogy([infor_rngd.time], [infor_rngd.disttoopt], '-x','MarkerSize',ms, 'LineWidth',lw, 'color', colors(8,:)); hold on;
    semilogy([infor_rnfr.time], [infor_rnfr.disttoopt], '-diamond','MarkerSize',ms, 'LineWidth',lw, 'color', colors(9,:)); hold on;
    semilogy([infor_rntgda.time], [infor_rntgda.disttoopt], '-square','MarkerSize',ms, 'LineWidth',lw, 'color', colors(10,:)); hold on;
    hold off;
    ax = gca;
    lg = legend({'RGDA', 'TSRGDA', 'RCEG', 'RHM', 'RCON', 'RFR', 'RTGDA', 'RNGD', 'RNFR', 'RNTGDA'}, 'NumColumns',2);
    lg.FontSize = 14;
    xlabel(ax,'Time','FontSize',22);
    ylabel(ax,'Optimality gap','FontSize',22);
    
    
    % gradnorm
    h3 = figure(3);
    semilogy([info_rgda.time], [info_rgda.gradnorm], '->','MarkerSize',ms, 'LineWidth',lw, 'color', colors(1,:)); hold on;
    semilogy([info_tsrgda.time], [info_tsrgda.gradnorm], '-*','MarkerSize',ms, 'LineWidth',lw, 'color', colors(2,:)); hold on;
    semilogy([info_rceg.time], [info_rceg.gradnorm], '-o','MarkerSize',ms, 'LineWidth',lw, 'color', colors(3,:)); hold on;
    semilogy([info_rhg_sdf.time], [info_rhg_sdf.gradnormf], '-d','MarkerSize',ms, 'LineWidth',lw, 'color', colors(4,:)); hold on;
    semilogy([info_rhg_con.time], [info_rhg_con.gradnormf], '-+','MarkerSize',ms, 'LineWidth',lw, 'color', colors(5,:)); hold on;
    semilogy([infor_rfr.time], [infor_rfr.gradnorm], '-s','MarkerSize',ms, 'LineWidth',lw, 'color', colors(6,:)); hold on;
    semilogy([infor_rtgda.time], [infor_rtgda.gradnorm], '-^','MarkerSize',ms, 'LineWidth',lw, 'color', colors(7,:)); hold on;
    semilogy([infor_rngd.time], [infor_rngd.gradnorm], '-x','MarkerSize',ms, 'LineWidth',lw, 'color', colors(8,:)); hold on;
    semilogy([infor_rnfr.time], [infor_rnfr.gradnorm], '-diamond','MarkerSize',ms, 'LineWidth',lw, 'color', colors(9,:)); hold on;
    semilogy([infor_rntgda.time], [infor_rntgda.gradnorm], '-square','MarkerSize',ms, 'LineWidth',lw, 'color', colors(10,:)); hold on;
    hold off;
    ax = gca;
    lg = legend({'RGDA', 'TSRGDA', 'RCEG', 'RHM', 'RCON', 'RFR', 'RTGDA', 'RNGD', 'RNFR', 'RNTGDA'}, 'NumColumns',2);
    lg.FontSize = 14;
    xlabel(ax,'Time','FontSize',22);
    ylabel(ax,'Gradnorm','FontSize',22);
    
    
   

    %% helper function
    % helper for setting parameters
    function options = set_options(model)
        % common options 
        options.update = 'exp';
        options.maxiter = maxiter;

        options.inv_method = 'tscg';
        
        switch setup
            case 1
                if strcmp(model, 'RGDA')
                    options.stepsize_x = 0.001;
                    options.stepsize_y = options.stepsize_x;
                    options.statsfun = @computestats;
                elseif strcmp(model, 'TSRGDA')
                    options.stepsize_x = 0.0005;
                    options.stepsize_y = 6; 
                    options.statsfun = @computestats;
                elseif strcmp(model, 'RHM')
                    options.stepsize = 0.05;
                    options.gamma = 0;
                    options.method = 'RH-con-fixedstep';
                    options.statsfun = @computestatsRHM;
                elseif strcmp(model, 'RCON')    
                    options.stepsize = 0.05;
                    options.gamma = 4;
                    options.method = 'RH-con-fixedstep';
                    options.statsfun = @computestatsRHM;
                elseif strcmp(model, 'RCEG') 
                    options.logchoice = 'log';
                    options.stepsize = 0.001;
                    options.statsfun = @computestats;
                elseif strcmp(model, 'RFR')
                    options.mode = 'fr';
                    options.stepsize_x = 0.001;
                    options.stepsize_y = 12;
                    options.epscg = 0;
                    options.statsfun = @computestats;
                elseif strcmp(model, 'RTGDA')
                    options.mode = 'tgda';
                    options.stepsize_x = 0.001;
                    options.stepsize_y = 12;
                    options.epscg = 0;
                    options.statsfun = @computestats;
                elseif strcmp(model, 'RNGD')
                    options.mode = 'ngd';
                    options.stepsize_x = 0.003;
                    options.stepsize_y = 0.8;
                    options.epscg = 0;
                    options.statsfun = @computestats;
                elseif strcmp(model, 'RNFR')
                    options.mode = 'nfr';
                    options.stepsize_x = 0.003;
                    options.stepsize_y = 0.8;
                    options.zeta = 1;
                    options.epscg = 0;
                    options.statsfun = @computestats;
                elseif strcmp(model, 'RNTGDA')
                    options.mode = 'ntgda';
                    options.stepsize_x = 0.003;
                    options.stepsize_y = 0.8;
                    options.zeta = 1;
                    options.epscg = 0;
                    options.statsfun = @computestats;
                end

        end
                    
    end


   


end

