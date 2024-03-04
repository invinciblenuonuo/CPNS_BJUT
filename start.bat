@echo off & cd /d %~dp0
mode con cols=100
call :showLogo
pause
call D:\ProgramData\anaconda3\Scripts\activate.bat D:\ProgramData\anaconda3
call conda activate qcar
cd /d D:\Documentes\postgraduate\qcar\official\CPNS_BJUT\official_code
python Traffic_Lights_Competition.py
pause



:showLogo
if not exist .outlogo (
echo CiAgICAgICAgICAgICBfX19fX18gLl9fX19fXyAgIC5fXyAgIF9fLiAgICAgIF9fX19fX18uICAgIF9fICAgICAgICAgIF9fXyAgICAgIC5fX19fX18gICAgICAgICAgICAgIAogICAgICAgICAgICAvICAgICAgfHwgICBfICBcICB8ICBcIHwgIHwgICAgIC8gICAgICAgfCAgIHwgIHwgICAgICAgIC8gICBcICAgICB8ICAgXyAgXCAgICAgICAgICAgICAKIF9fX19fXyAgICB8ICAsLS0tLSd8ICB8XykgIHwgfCAgIFx8ICB8ICAgIHwgICAoLS0tLWAgICB8ICB8ICAgICAgIC8gIF4gIFwgICAgfCAgfF8pICB8ICAgICBfX19fX18gCnxfX19fX198ICAgfCAgfCAgICAgfCAgIF9fXy8gIHwgIC4gYCAgfCAgICAgXCAgIFwgICAgICAgfCAgfCAgICAgIC8gIC9fXCAgXCAgIHwgICBfICA8ICAgICB8X19fX19ffAogICAgICAgICAgIHwgIGAtLS0tLnwgIHwgICAgICB8ICB8XCAgIHwgLi0tLS0pICAgfCAgICAgIHwgIGAtLS0tLi8gIF9fX19fICBcICB8ICB8XykgIHwgICAgICAgICAgICAKICAgICAgICAgICAgXF9fX19fX3x8IF98ICAgICAgfF9ffCBcX198IHxfX19fX19fLyAgICAgICB8X19fX19fXy9fXy8gICAgIFxfX1wgfF9fX19fXy8gICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAo= >.logo
certutil -decode .logo .outlogo>nul
del .logo
)
type .outlogo
goto :eof

