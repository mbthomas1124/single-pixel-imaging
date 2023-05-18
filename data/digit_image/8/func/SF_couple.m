function y= SF_couple(signal, pump, coupling_mode,lamd_s,lamd_p,lamd_sf,Npxl,Npxl_s,dx,L,f,kai)

[fsf_out,fs_out,fp_out]=SFG_300(signal,pump,lamd_s,lamd_p,Npxl_s,dx,L,kai);        %     fs_ref neigbot reference , here is gaussian with new LG01         *
fsf_coll=SFG_lens_2f(fsf_out,f,L,lamd_sf,dx,Npxl,Npxl_s);

conv=abs(sum(sum(fsf_coll.*coupling_mode))).^2/sum(sum(abs(coupling_mode).^2));

y=conv;