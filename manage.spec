# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['C:\\Users\\GokulKannan\\Downloads\\people_counter\\people_counter\\manage.py'],
             pathex=['C:\\Users\\GokulKannan\\Downloads\\people_counter'],
             binaries=[],
             datas=[('C:\\Users\\GokulKannan\\Downloads\\people_counter\\static','static'),
             ('C:\\Users\\GokulKannan\\Downloads\\people_counter\\viewer\\migrations','viewer\\migrations'),
             ('C:\\Users\\GokulKannan\\Downloads\\people_counter\\viewer\\models','viewer\\models'),
             ('C:\\Users\\GokulKannan\\Downloads\\people_counter\\viewer\\static','viewer\\static'),
             ('C:\\Users\\GokulKannan\\Downloads\\people_counter\\viewer\\templates','viewer\\templates'),
             ('C:\\Users\\GokulKannan\\Downloads\\people_counter\\viewer\\videos','viewer\\videos')],
             hiddenimports=['people_counter.urls','viewer.urls','scipy.special.cython_special'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='manage',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='manage')
