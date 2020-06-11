/* Written by Matyas Sustik based on original code written by Zoltan
 * Hidvegi.
 *
 * No warranty of any kind provided: Intended for personal and research
 * use only.
 *
 * $Id: startgdb.h,v 1.1 2011-05-12 03:31:57 sustik Exp $
 */

#include <sys/types.h>
#include <sys/wait.h>
#include <signal.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

void static startgdb(void)
{
    char* env;
    char srcpath[1024];
    char pid[20];
    static long dpid = 0;
    char buf[4096];

    if (dpid) {
        int status;
        pid_t wpid = waitpid(dpid, &status, WNOHANG);
        if (wpid != dpid && !kill(dpid, 0))
            return;
    }
    sprintf(pid, "%ld", (long) getpid());
    if ((env = getenv("GDEBUG")))
        sprintf(srcpath, "%s", env);
    else
        sprintf(srcpath, ".");
    if (!(env = getenv("DISPLAY"))) { // Non X-Windows mode:
        long child;
        child = fork();
        if (!child) {
            dpid = getpid();
            setpgid(0, 0);
        } else {
            sprintf(pid, "%ld", child);
            sleep(1);
            execlp("gdb", "gdb", "-dir", srcpath, "/dev/null", pid, NULL);
            perror("Error attaching gdb");
            kill(child, SIGKILL);
            return;
        }
    } else {                          // X-Windows mode:
        dpid = fork();
        if (!dpid) {
            char geometry[20];
            setpgid(0, 0);
            sprintf(buf, "exec gdb -dir '%s' '%s' %s", srcpath, "/dev/null",
		    pid);
            if ((env = getenv("GDB_GEOMETRY"))) {
                sprintf(geometry, "%s", env);
		execlp("xterm", "xterm ", "-title", "gdb", "-geometry",
		       geometry, "-sb", "-sl", "2012", "-fn", "10x20",
		       "-e", "sh", "-c", buf, NULL);
	    } else {
		execlp("xterm", "xterm ", "-title", "gdb", "-sb", "-sl",
		       "2012", "-fn", "10x20", "-e", "sh", "-c", buf, NULL);
		perror("Error attaching xterm/gdb");
		_exit(1);
	    }
	}
    }
    sleep(10);
}
