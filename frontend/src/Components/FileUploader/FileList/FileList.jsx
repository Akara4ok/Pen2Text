import React from 'react';
import classes from './FileList.scss';

class FileList extends React.Component {
    constructor(props) {
        super(props);
    }

    render() {
        const { children } = this.props;
        return <div className={classes.wrapper}>{children}</div>;
    }
}

export default FileList;
