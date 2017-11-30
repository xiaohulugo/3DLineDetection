/* --- --- ---
 * Copyright (C) 2008--2010 Idiap Research Institute (.....@idiap.ch)
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. The name of the author may not be used to endorse or promote products
 *    derived from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
// Timers.h: interface for the CTimer class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(_TIMER_H_)
#define _TIMER_H_

#include <ctime>						// clock
#include "time.h"
#include <math.h>

#ifndef CLOCKS_PER_SEC                          /* define clocks-per-second if needed */
#define CLOCKS_PER_SEC 1000000
#endif

class CTimer
{
public:
	/* set the starting time */
	void Start() {
	    startTime = clock();
        startLocalTime = time(0);
    };

	/* stop the time */
	void Stop(bool local_time=false) {
		stopTime = clock();
        stopLocalTime = time(0);

        if ( local_time )
            elapsedTime = difftime(stopLocalTime, startLocalTime);
        else
            elapsedTime = (stopTime - startTime) / (double)(CLOCKS_PER_SEC);

		if ( elapsedTime <= 0 )
            elapsedTime = (stopTime - startTime) / (double)(CLOCKS_PER_SEC);
	};

	/* get the elapsed hours */
	double GetElapsedHours() { return elapsedTime/3600.0;  }

	/* get the elapsed minutes */
	double GetElapsedMinutes() { return elapsedTime/60.0;  }

	/* get the elapsed seconds */
	double GetElapsedSeconds() { return elapsedTime; }

	/* print the elapsed time */
	void PrintElapsedTimeMsg(char* msg,
			bool used_hours = true,
			bool used_minutes = true,
			bool used_seconds = true) {
		if ( !msg )  return;

		if ( used_hours && used_minutes && used_seconds ) {
			int n_hours = (int)GetElapsedHours();
			if ( n_hours > 0 ) {
				int n_minutes = (int)fmod(GetElapsedMinutes(),60.0);
				if ( n_minutes > 0 )
					sprintf(msg, "%d hours %d minutes %2.4f seconds",
						n_hours, n_minutes, fmod(GetElapsedSeconds(),60.0) );
				else
					sprintf(msg, "%d hours %2.4f seconds",
						n_hours, fmod(GetElapsedSeconds(),60.0) );
			}
			else {
				int n_minutes = (int)fmod(GetElapsedMinutes(),60.0);
				if ( n_minutes > 0 )
					sprintf(msg, "%d minutes %2.4f seconds",
						n_minutes, fmod(GetElapsedSeconds(),60.0) );
				else
					sprintf(msg, "%2.4f seconds", GetElapsedSeconds() );
			}
		}
		else if ( used_hours && used_minutes ) {
			int n_hours = (int)GetElapsedHours();
			if ( n_hours > 0 )
				sprintf(msg, "%d hours %2.4f minutes",
					n_hours, fmod(GetElapsedMinutes(),60.0) );
			else
				sprintf(msg, "%2.4f minutes", GetElapsedMinutes() );
		}
		else if ( used_hours && used_seconds ) {
			int n_hours = (int)GetElapsedHours();
			if ( n_hours > 0 )
				sprintf(msg, "%d hours %2.4f minutes",
					n_hours, fmod(GetElapsedSeconds(),3600.0) );
			else
				sprintf(msg, "%0.4f seconds", GetElapsedSeconds() );
		}
		else if ( used_minutes && used_seconds ) {
			int n_minutes = (int)GetElapsedMinutes();
			if ( n_minutes > 0 )
				sprintf(msg, "%d minutes %0.4f seconds",
					n_minutes, fmod(GetElapsedSeconds(),60.0) );
			else
				sprintf(msg, "%0.4f seconds", GetElapsedSeconds() );
		}
		else if ( used_hours )
			sprintf(msg, "%.4f hours", GetElapsedHours() );
		else if ( used_minutes )
			sprintf(msg, "%.4f minutes", GetElapsedMinutes() );
		else if ( used_seconds )
			sprintf(msg, "%.4f seconds", GetElapsedSeconds() );
	};


    /* print now local global time using strftime
       format_args example: "Now is %Y-%m-%d %H:%M:%S" */
    void PrintLocalTime(char* buffer, int length, const char* format_args) {
        time_t rawtime;
        struct tm * timeinfo;

        time ( &rawtime );
        timeinfo = localtime ( &rawtime );

        strftime (buffer, length, format_args, timeinfo);
    };

	CTimer() { elapsedTime = 0; };
	virtual ~CTimer() {};

private:
    time_t startLocalTime;
    time_t stopLocalTime;

	/* starting time */
	clock_t startTime;
	clock_t stopTime;
	double elapsedTime;
};

#endif // !defined(_TIMER_H_)
