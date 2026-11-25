export const shapes={wall:{type:'box',size:[.022,.115,.10]},pillar:{type:'cylinder',size:[.042,.115]},sphere:{type:'sphere',size:[.055]}};
export const materials={
 rigid:{solref:[.004,1],solimp:[.95,.99,.001,.5,2],color:0xc86b42,opacity:.92,roughness:.4},
 soft:{solref:[.07,1.6],solimp:[.12,.85,.05,.5,2],color:0x8c65c6,opacity:.64,roughness:.94},
 bouncy:{solref:[.035,.08],solimp:[.55,.96,.02,.5,2],color:0xe5a72b,opacity:.78,roughness:.23},
};
export const obstacleHome=[.42,.15,.23];
