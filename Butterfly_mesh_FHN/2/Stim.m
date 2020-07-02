function Istim = Stim(stimulation, time, condition)
    if time < stimulation.period
        Istim = 0;
    else
        beat= mod(stimulation.period, time);
        if beat >0 & beat < stimulation.duration
            if condition == 'zero'
                Istim = 0;
            elseif condition =='regular'
                Istim = abs(stimulation.amplitude);
            end
        end 
    end
end