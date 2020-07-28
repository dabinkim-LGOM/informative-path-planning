% clear all;
% close all;

hotspot = 'hotspot_info'; info = 'info_gain';  mean = 'mean'; mse = 'MSE'; UCB = 'UCB'; MVI = 'MVI';
metric = {hotspot, info, mean, mse, UCB};
x_metric = size(metric);

env_type_list = {'Free', 'Box', 'Harsh'};
reward_list = {'info_gain', 'mean'};
% iter_list = {'0', '1', '2', '3', '4'};
% grad_list = {'0.0', '0.05', '0.1', '0.15', '0.2'};
iter_list = {'1'};
grad_list = {'0.0', '0.1'};
x1 = size(iter_list);
x2 = size(grad_list);
data = {};

for grad_nnn = 1:x2(2)
    for iter_nnn = 1:x1(2)
        iter = iter_list{iter_nnn};
        grad_step = grad_list{grad_nnn};
        range = '100.0';
        
        param =  {'grad_step', str2num(grad_step);
            'range', str2num(range)};
        data{iter_nnn}.param = param;
        data{iter_nnn}.time = (1:1:150)';
        
        for i=1:x_metric(2)
%             dir mean/Free
%             dir Harsh
%             directory = fullfile(fullfile(pwd, reward_list(1)), env_type_list(2));
            directory = '';
            filename = strcat(directory,'metrics_grad_',...
                num2str(grad_step), 'range_max_', num2str(range),...
                ' iter_', num2str(iter), '_', metric{i}, '.txt');
            disp(filename);
            s  = importdata(filename);
            data{iter_nnn} = setfield(data{iter_nnn}, metric{i}, s);
        end
        
    end
    
    mean_data.param = data{1}.param;
    mean_data.time = data{1}.time;
    hotspot_mean = zeros(length(mean_data.time),1);
    info_mean = zeros(length(mean_data.time),1);
    mean_mean = zeros(length(mean_data.time),1);
    MSE_mean = zeros(length(mean_data.time),1);
%     MSE_mean = data{1}.MSE(1:length(mean_data.time));
    UCB_mean = zeros(length(mean_data.time),1);
    
    for i=1:length(mean_data.time)
        for j=1:length(data)
            hotspot_mean(i) = (hotspot_mean(i)*(j-1) + data{j}.hotspot_info(i))/j ;
            info_mean(i) = (info_mean(i)*(j-1) + data{j}.info_gain(i))/j ;
            mean_mean(i) = (mean_mean(i)*(j-1) + data{j}.mean(i))/j ;
            MSE_mean(i) = (MSE_mean(i)*(j-1) + data{j}.MSE(i))/j ;
            UCB_mean(i) = (UCB_mean(i)*(j-1) + data{j}.UCB(i))/j ;
        end
    end
    
    mean_data.hotspot_info = hotspot_mean;
    mean_data.info_gain = info_mean;
    mean_data.mean = mean_mean;
    mean_data.MSE = MSE_mean;
    mean_data.UCB = UCB_mean;
    
    figure(1); 
    subplot(2,3,1); hold on;
    plot(mean_data.time, mean_data.hotspot_info);
    xlabel('Timestep', 'FontSize',16); ylabel('Hotspot Info.', 'FontSize',16);
    legend('\eta=0.0', '\eta=0.05', '\eta=0.10', '\eta=0.15', '\eta=0.20');
    ax = gca; ax.FontSize =16;
    subplot(2,3,2); hold on;
    plot(mean_data.time, mean_data.info_gain);
    xlabel('Timestep', 'FontSize',16); ylabel('Information Gain', 'FontSize',16);
    legend('\eta=0.0', '\eta=0.05', '\eta=0.10', '\eta=0.15', '\eta=0.20');
    ax = gca; ax.FontSize =16;

    subplot(2,3,3); hold on;
    plot(mean_data.time, mean_data.mean);
    xlabel('Timestep', 'FontSize',16); ylabel('Mean Gain', 'FontSize',16);
    legend('\eta=0.0', '\eta=0.05', '\eta=0.10', '\eta=0.15', '\eta=0.20');
    ax = gca; ax.FontSize =16;
    
    subplot(2,3,4); hold on;
    plot(mean_data.time, mean_data.MSE);
    xlabel('Timestep', 'FontSize',16); ylabel('MSE', 'FontSize',16);
    legend('\eta=0.0', '\eta=0.05', '\eta=0.10', '\eta=0.15', '\eta=0.20');
    ax = gca; ax.FontSize =16;
    
    subplot(2,3,5); hold on;
    plot(mean_data.time, mean_data.UCB);
    xlabel('Timestep', 'FontSize',16); ylabel('GP-UCB', 'FontSize',16);
    legend('\eta=0.0', '\eta=0.05', '\eta=0.10', '\eta=0.15', '\eta=0.20');
    ax = gca; ax.FontSize =16;
end
