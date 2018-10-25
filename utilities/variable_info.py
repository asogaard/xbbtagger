default_values = { 
	"MV2c10_discriminant"	: -99,
	"MV2c10mu_discriminant"	: -99,
	"MV2c10rnn_discriminant"	: -99,
	"IP2D_pu"	: -99,
	"IP2D_pc"	: -99,
	"IP2D_pb"	: -99,
	"IP3D_pu"	: -99,
	"IP3D_pc"	: -99,
	"IP3D_pb"	: -99,
	"rnnip_pu"	: -99,
	"rnnip_pc"	: -99,
	"rnnip_pb"	: -99,
	"rnnip_ptau"	: -99,
	'SV1_pu'	: -99,
	'SV1_pc'	: -99,
	'SV1_pb'	: -99,
	"JetFitter_energyFraction"	: -99,
	"JetFitter_mass"	: -99,
	"JetFitter_significance3d"	: -99,
	"JetFitter_deltaphi"	: -99,
	"JetFitter_deltaeta"	: -99,
	"JetFitter_massUncorr"	: -99,
	"JetFitter_dRFlightDir"	: -99,
	"JetFitter_nVTX"	: -1,
	"JetFitter_nSingleTracks"	: -99,
	"JetFitter_nTracksAtVtx"	: -1,
	"JetFitter_N2Tpair"	: -1,
	"SV1_masssvx"	: -99,
	"SV1_efracsvx"	: -99,
	"SV1_significance3d"	: -1,
	"SV1_dstToMatLay"	: -99,
	"SV1_deltaR"	: -1,
	"SV1_Lxy"	: -1,
	"SV1_L3d"	: -1,
	"SV1_N2Tpair"	: -1,
	"SV1_NGTinSvx"	: -99,
#	"deta"	: -99,
#	"dphi"	: -99,
#	"dr"	: -99,
	"pt"	: -99,
	"eta"	: -99,
	"mass"	: -99
}



default_vars = [
	"MV2c10_discriminant"
	#"MV2c10mu_discriminant",
	#"MV2c10rnn_discriminant"
]

prob_vars = [
	"IP2D_pu",
	"IP2D_pc",
	"IP2D_pb",
	"IP3D_pu",
	"IP3D_pc",
	"IP3D_pb",
	"rnnip_pu",
	"rnnip_pc",
	"rnnip_pb",
	"rnnip_ptau",
	'SV1_pu',
	'SV1_pc',
	'SV1_pb'
]

jetfitter_vars = [
	"JetFitter_energyFraction",
	"JetFitter_mass",
	"JetFitter_significance3d",
	"JetFitter_deltaphi",
	"JetFitter_deltaeta",
	"JetFitter_massUncorr",
	"JetFitter_dRFlightDir",
	"JetFitter_nVTX",
	"JetFitter_nSingleTracks",
	"JetFitter_nTracksAtVtx",
	"JetFitter_N2Tpair"
]

SV1_vars = [
	"SV1_masssvx",
	"SV1_efracsvx",
	"SV1_significance3d",
	"SV1_dstToMatLay",
	"SV1_deltaR",
	"SV1_Lxy",
	"SV1_L3d",
	"SV1_N2Tpair",
	"SV1_NGTinSvx"
]
	
kin_vars = [
	"pt",
	"eta"
]

#angular_vars = [
#	"deta",
#	"dphi",
#	"dr"
#]

fat_jet_vars = [
	"pt",
	"eta",
	"mass",
]
