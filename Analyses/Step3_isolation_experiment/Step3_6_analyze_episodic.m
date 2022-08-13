% load Step5_high_exp_20_e_avoiding_zero
% 
% 'Much experience'
% 
% Highexpgroup.log_Revisitrates=log10(Highexpgroup.Revisitrates+0.01);
% find_non0=find(Highexpgroup.Revisitrates~=0);
% fitglme(Highexpgroup(find_non0,:),'log_Revisitrates ~ Fruit + Fruit*Nights + (1|Batname)')


clear

Highexpgroup = readtable('Step3_sta_2——2_grouped_be_in_all_matlab.csv');

Highexpgroup.log_Revisitrates=log10(Highexpgroup.Revisitrates+0.01);
find_non0=find(Highexpgroup.Revisitrates~=0);
fitglme(Highexpgroup(find_non0,:),'log_Revisitrates ~ Fruit + Nights + Fruit*Nights + Distance + (1|Batname)')
