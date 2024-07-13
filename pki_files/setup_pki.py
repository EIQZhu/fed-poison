from Cryptodome.PublicKey import ECC

# generate client keys公钥g^ai
for i in range (10):
	key = ECC.generate(curve='P-256')
	hdr = 'client'+str(i)+'.pem'
	f = open(hdr, 'wt')
	f.write(key.export_key(format='PEM'))
	f.close()