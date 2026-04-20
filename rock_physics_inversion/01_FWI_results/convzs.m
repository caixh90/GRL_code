function [ fw ] = convzs( f , w )
%Scott's simple convz (no weird edge issues)

fw_big = conv(f,w);
fw = fw_big(ceil(length(w)/2):ceil(length(w)/2)+length(f)-1);

end

