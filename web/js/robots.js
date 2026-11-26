export const DEFAULT_ROBOT='openyam';
export const robots={
 openyam:{id:'openyam', name:'OpenYAM',tag:'OPENYAM', checkpoint:'e6-submm@20k',checkpointLabel:'E6 sub-mm @ 20k',
  assets:'robots/openyam',home:[0,1.047,1.047,0,0,0],kp:[50,60,50,25,15,10],kd:[3,4,3,2,1,.8],peak:[27,27,27,7,7,7],
  payloads:[0,.5,1,2],payloadModel:'TCP-mounted 40 mm cube',camera:[.95,-1.1,.75],view:[.13,0,.17],obstacle:[.42,.15,.23],
  fingers:true,tcpBody:'arm_tcp',elbow:'arm_link_3',profiles:{
   reach:{a:[.025,-.13,.17,.035,-.045,.025],b:[0,.025,.015,-.02,.015,0],n:[1,1,1,1,1,1]},
   sweep:{a:[.23,.025,-.035,.04,.065,.04],b:[0,.045,.035,.015,.02,.015],n:[1,2,2,1,2,1]},
   orbit:{a:[.17,-.09,.11,.05,-.035,.04],b:[0,.065,.055,.025,.025,.02],n:[1,1,1,1,1,1]}
  }},
 'metal-arm':{id:'metal-arm',name:'Metal Arm',tag:'METAL ARM',checkpoint:'metal-stage2@20k',checkpointLabel:'Metal Arm Stage 2 @ 20k',
  assets:'robots/metal-arm',home:[0,-1.05,1.45,-.4,0,0],kp:[300,500,500,400,25,20],kd:[5,5,5,4,.2,.2],peak:[30,100,30,10,10,10],
  payloads:[0,.5,1,2,3],payloadModel:'Centered rigid TCP sphere, 20 mm radius, merged hand inertia. Simulation payload is not a hardware load rating.',
  camera:[.95,-.755,.84],view:[.13,.345,.26],obstacle:[.42,.495,.32],tcpBody:'hand',tcpSite:'arm_tcp_site',elbow:'arm_link_3',profiles:{
   reach:{a:[.025,-.13,.17,.035,-.045,.025],b:[0,.025,.015,-.02,.015,0],n:[1,1,1,1,1,1]},
   sweep:{a:[.23,.025,-.035,.04,.065,.04],b:[0,.045,.035,.015,.02,.015],n:[1,2,2,1,2,1]},
   orbit:{a:[.17,-.09,.11,.05,-.035,.04],b:[0,.065,.055,.025,.025,.02],n:[1,1,1,1,1,1]}
  }},
};
export const robotId=value=>Object.hasOwn(robots,value)?value:DEFAULT_ROBOT;
