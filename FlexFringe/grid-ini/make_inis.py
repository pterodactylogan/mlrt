import os
def make_ini_file(performfirst, reversetraces, extend, shallowfirst, depthcheck, symbolcheck, search, sinkson):
	lines = ["[default]", "heuristic-name = evidence_driven", "data-name = edsm_data", ]
	lines.append(f"performfirst = {performfirst}")
	lines.append(f"reversetraces = {reversetraces}")
	lines.append(f"extend = {extend}")
	lines.append(f"shallowfirst = {shallowfirst}")
	lines.append(f"depthcheck = {depthcheck}")
	lines.append(f"symbolcheck = {symbolcheck}")
	if search != "none":
		lines.append(f"{search} = 1")
	lines.append(f"sinkson = {sinkson}")
	
	file_path = f"{performfirst}.{reversetraces}.{extend}.{shallowfirst}.{depthcheck}.{symbolcheck}.{search}.{sinkson}"
	print(file_path)
	if sinkson:
		for sinkcount in [1, 10, 25]:
			for mergesinkscore in [0, 1]:
				new_file_path = file_path + f".{sinkcount}.{mergesinkscore}"
				with open(f"{new_file_path}.ini", "a") as f:
					for line in lines: 
						f.write(f"{line}\n")
					f.write(f"sinkcount = {sinkcount}\n")
					f.write(f"mergesinkscore = {mergesinkscore}\n")
				f.close()
	else: 
		with open(f"{file_path}.ini", "a") as f:
			for line in lines:
				f.write(f"{line}\n")
		f.close() 

for performfirst in [0, 1]:
	for reversetraces in [0, 1]:
		for extend in [0, 1]:
			for shallowfirst in [0, 1]:
				for depthcheck in [0, 1]:
					for symbolcheck in [0, 1]:
						for search in ["searchdeep", "searchlocal", "searchglobal", "searchpartial", "none"]:
							for sinkson in [0, 1]:
								make_ini_file(performfirst, reversetraces, extend, shallowfirst, depthcheck, symbolcheck, search, sinkson)
	
