[CmdletBinding()]
param (
	[uint32] [alias("run-id", "id")] $run_id,
	[string] [alias("o")] $outfile  = "auto",
	[string] $tmpdir = "frames",
	[string] $rundir = "runs",
	[uint64] $pad    = 5,
	[string]
		[validateset("size", "speed", "balanced", "aom-av1", "x264", "x265")]
		$optimize = "speed",
	[switch] $dontopen,
	[switch] $keepframes,
	[switch] $help
)

if ($help) {
	get-help $MyInvocation.MyCommand.source
	exit 0
}

$file = "$run_id.npz"

if ($outfile -eq "auto") {
	$outfile = "vids/$run_id.mp4"
}

$script = @"
from common import cv2, load_runs, Path
file = "$file"

try:
`tframes = load_runs(file, runs_dir="$rundir")
except ValueError:
`texit(1)

Path("$tmpdir").mkdir(exist_ok=True)

print("extracting frame \x1b7", end="", flush=True)
total = len(frames)

for i, x in enumerate(frames):
	if i % 107 == 0:
		print(f"\x1b8{i:,}/{total:,}", end="", flush=True)

	cv2.imwrite(f"$tmpdir/{i:0${pad}d}.png", x[0])

print(f"\rextracted {total:,} frames\x1b[K")
"@


if (-not (test-path -type leaf $outfile)) {
	if (-not (gcm -type app -ea ignore python)) {
		throw "required program ``python`` was not found"
	}

	if (-not (gcm -type app -ea ignore ffmpeg)) {
		throw "required program ``ffmpeg`` was not found"
	}

	[uint32] $fps = python -c "from common import fps; print(fps)"

	if (-not (test-path -type container $tmpdir)) {
		python -c $script

		if ($lastExitCode -eq 1) {
			write-host "invalid run id. file not found"
			exit 1
		}
	}

	$flags = switch ($optimize) {
		# I tested a bunch of codecs and presets, and these ones were basically the best
		"size"     {"-c:v", "libx265", "-preset", "ultrafast"}
		"speed"    {"-c:v", "libx264", "-preset", "ultrafast"}
		"balanced" {"-c:v", "libx264", "-preset", "superfast"}

		"aom-av1"  {"-c:v", "libaom-av1"}
		"x264"     {"-c:v", "libx264"}
		"x265"     {"-c:v", "libx265"}
	}
}

ffmpeg -hide_banner -framerate $fps -i "$tmpdir/%0${pad}d.png" -pix_fmt gray `
	-vf "scale=1360:850:flags=neighbor" @flags $outfile

if (-not $keepframes.ispresent -and (test-path -type container $tmpdir)) {
	write-host "deleting frame images"
	rm -recurse -force $tmpdir
}

if (-not $dontopen.ispresent) {
	write-host "opening " -nonewline
	start $outfile
}

write-host "file: $outfile"

exit 0
