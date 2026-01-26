@echo off
echo Backing up database...
:: Create backup folder if not exists
if not exist "backup" mkdir backup

:: Generate date string
set d=%date:~0,4%%date:~5,2%%date:~8,2%
if "%d%"=="" set d=backup_file

echo Copying database to backup folder...
xcopy "data\attendance.db" "backup\attendance_%d%.db*" /Y

echo Backup Completed!
timeout /t 3