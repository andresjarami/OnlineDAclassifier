clear all
for person=1:1:120    
    disp(person)
    user=join(['user',num2str(person)]);   
        
    load(strcat('allUsers_data\rawData\',user,'\userData.mat'));
    
    names = {'training' 'testing'};
    rpt_1=0;
    rpt_2=0;
    rpt_3=0;
    rpt_4=0;
    rpt_5=0;
    rpt_6=0;
    for name_idx = 1:2
        for aux_rpt=1:130
            
            if strcmp(names{name_idx},'training')
                rawEmg=userData.training{aux_rpt,1}.emg;
                label=userData.training{aux_rpt,1}.label;
            elseif strcmp(names{name_idx},'testing')
                rawEmg=userData.training{aux_rpt,1}.emg;
                label=userData.training{aux_rpt,1}.label;
            end
            emg=segmentationGesture(rawEmg);
            
            %plot(1:length(rawEmg),rawEmg(:,1))
            %figure;plot(1:length(emg),emg(:,1))
            
            if strcmp(label,'waveIn')
                class=1;
                rpt_1=rpt_1+1;
                rpt=rpt_1;
            elseif strcmp(label,'waveOut')
                class=2;
                rpt_2=rpt_2+1;
                rpt=rpt_2;
            elseif strcmp(label,'fist')
                class=3;
                rpt_3=rpt_3+1;
                rpt=rpt_3;
            elseif strcmp(label,'fingersSpread')
                class=4;
                rpt_4=rpt_4+1;
                rpt=rpt_4;
            elseif strcmp(label,'doubleTap')
                class=5;
                rpt_5=rpt_5+1;
                rpt=rpt_5;
            elseif strcmp(label,'relax')
                class=6;
                rpt_6=rpt_6+1;
                rpt=rpt_6;
            end
            
            save(strcat('segmented_data\emg_person',...
                num2str(person),'_class',num2str(class),'_rpt',num2str(rpt)),'emg');         
            
            
        end
    end
            
end