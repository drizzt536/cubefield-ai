// go build -trimpath -ldflags="-s -w" server.go

package main

import (
	"net/http"
	"time"
	"flag"
	"os"
)

var done = make(chan bool)

func main() {
	var (
		ip      string
		port    string
		persist bool
	)

	{
		var ip_arg      *string = flag.String("ip"     , "127.0.0.1", "server IP address"          )
		var port_arg    *string = flag.String("port"   , "80"       , "server http port"           )
		var persist_arg *bool   = flag.Bool  ("persist", false      , "persist after serving files")

		flag.Parse()

		ip      = *ip_arg
		port    = *port_arg
		persist = *persist_arg
	}

	// I don't particularly care that raw HTTP clients can have a GET for
	// stuff like `/../something`. this is intended to be locally.
	http.HandleFunc("/", func (w http.ResponseWriter, r *http.Request) {
		var file string = r.URL.Path[1:]
		if file == "" {
			file = "index.html"
		}

		if file == "kys" && !persist {
			println("exiting in 1s")
			time.Sleep(1 * time.Second)
			close(done)
			return
		}

		println("serving file '" + file + "'")

		http.ServeFile(w, r, file)
	})

	go func () {
		<- done
		os.Exit(0)
	}()

	var address string = ip + ":" + port

	println("waiting for a connection to http://" + address)
	err := http.ListenAndServe(address, nil)

	if err != nil {
		println("Server error: " + err.Error())
	}
}
