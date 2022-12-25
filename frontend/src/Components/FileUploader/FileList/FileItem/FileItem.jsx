import React from 'react';
import classes from './FileItem.scss';

class FileItem extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            id: this.props.id,
            index: this.props.index,
        };
    }
    render() {
        const { id, children, index } = this.props;
        return (
            <div className={classes.wrapper} id={id}>
                <span
                    onClick={() => {
                        this.props.setFileViewMode();
                        this.props.goToSelectedFile(index);
                    }}>
                    {children}
                </span>
                <button onClick={() => this.props.deleteByIndex(index)}>
                    &#10006;
                </button>
            </div>
        );
    }
}

export default FileItem;
