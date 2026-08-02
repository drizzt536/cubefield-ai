// go build -trimpath -ldflags="-s -w" server.go

package main

import (
	"path/filepath"
	"net/http"
	"os/exec"
	"time"
	"flag"
	"os"
)

func main() {
	var (
		ip      string
		port    string
		persist bool
		open    bool
	)

	{
		var ip_arg      *string = flag.String("ip"       , "127.0.0.1", "server IP address")
		var port_arg    *string = flag.String("port"     , "80"       , "server http port")
		var persist_arg *bool   = flag.Bool  ("persist"  , false      , "persist after serving files")
		var open_arg    *bool   = flag.Bool  ("no-launch", false      , "don't oepn the page in the browser")

		flag.Parse()

		ip      = *ip_arg
		port    = *port_arg
		persist = *persist_arg
		open    = !*open_arg
	}

	exePath, _ := os.Executable()
	exePath = filepath.Dir(exePath)

	// I don't particularly care that raw HTTP clients can have a GET for
	// stuff like `/../something`. this is intended to be locally anyway.
	http.HandleFunc("/", func (w http.ResponseWriter, r *http.Request) {
		var file string = r.URL.Path[1:]

		if file == "" {
			file = "index.html"
		} else if file == "kys" && !persist {
			println("exiting in 1s")
			time.Sleep(1 * time.Second)
			// there shouldn't be any stuff left partially in flight.
			// also, I don't care even if there is.
			os.Exit(0)
		}

		println("serving file '" + file + "'")
		file = filepath.Join(exePath, file)

		http.ServeFile(w, r, file)
	})

	var address string = ip + ":" + port

	full_address := "http://" + address

	if open {
		println("opening " + full_address)
		exec.Command("cmd", "/c", "start", full_address).Start()
	} else {
		println("waiting for a connection to " + full_address)
	}

	err := http.ListenAndServe(address, nil)

	if err != nil {
		println("Server error: " + err.Error())
	}
}
